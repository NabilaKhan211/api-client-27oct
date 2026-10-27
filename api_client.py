"""Simple API Client: current weather and a 3-day forecast from the Open-Meteo REST API.

Open-Meteo is a free public API that needs NO API KEY, so there is no secret
that could be exposed in the code or committed to GitHub.

Two GET requests are made:
    1. Geocoding API  - turns a city name into latitude and longitude
    2. Forecast API   - returns current weather and a daily forecast as JSON

Error handling (the point of this task):
    * the HTTP status code of every response is checked
    * connection failures and timeouts are caught (and retried a couple of times)
    * temporary problems (HTTP 429 and 5xx) are retried with a pause; permanent
      errors (400, 403, 404 ...) are reported straight away
    * the JSON itself is validated before it is used
    * every failure becomes a short, friendly message, never a traceback

Usage:
    python api_client.py Mumbai
    python api_client.py New York --raw        (also print the raw JSON response)
    python api_client.py Delhi --save          (save the raw responses to last_response.json)
    python api_client.py                       (interactive: type city names one after another)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:  # friendly message instead of a traceback
    print("The 'requests' library is not installed. Install it with:  pip install requests")
    sys.exit(1)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_TIMEOUT = (5, 10)      # seconds to connect, seconds to wait for the reply
MAX_RETRIES = 2                # extra attempts after the first one, for temporary problems only
BACKOFF_SECONDS = 1.0          # pause before retry 1; doubles each time (1s, 2s ...)
MAX_RETRY_WAIT = 10.0          # never wait longer than this, even if the server asks us to
SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_response.json")

CURRENT_VARIABLES = "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
DAILY_VARIABLES = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum"
FORECAST_DAYS = 3

# WMO weather interpretation codes used by Open-Meteo
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


# ---------- Errors: every message is safe and friendly to show to the user ----------
class ApiError(Exception):
    """Base class for every problem this client reports."""


class NetworkError(ApiError):
    """Could not reach the server (no internet, DNS failure, timeout ...)."""


class HttpStatusError(ApiError):
    """The server answered with an error status code."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class BadResponseError(ApiError):
    """The server answered 200 OK but the body was not what we expected."""


class CityNotFoundError(ApiError):
    """The geocoding API knows no city with that name."""


# ---------- Helpers ----------
def explain_status(status, reason=None):
    """Turn an HTTP status code into a clear sentence."""
    if status == 400:
        text = "The API rejected the request as invalid (HTTP 400)"
        return f"{text}: {reason}" if reason else text + "."
    if status in (401, 403):
        return f"Access to the API was denied (HTTP {status}). An API key or different permissions may be required."
    if status == 404:
        return "The API address or resource was not found (HTTP 404)."
    if status == 429:
        return "Too many requests: the API rate limit was reached (HTTP 429)."
    if status >= 500:
        return f"The API server had a problem (HTTP {status}). Please try again later."
    return f"The API returned an unexpected status (HTTP {status})."


def error_reason(response):
    """Open-Meteo explains errors as {"error": true, "reason": "..."}. Returns the reason or None."""
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict) and isinstance(body.get("reason"), str):
        return body["reason"][:200]
    return None


def retry_after_seconds(response):
    """Read the Retry-After header (in seconds). Returns None if absent or not a number."""
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return min(max(seconds, 0.0), MAX_RETRY_WAIT)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------- The client ----------
class ApiClient:
    """Makes the HTTP requests and checks everything that can go wrong."""

    def __init__(self, geocoding_url=GEOCODING_URL, forecast_url=FORECAST_URL, timeout=DEFAULT_TIMEOUT,
                 max_retries=MAX_RETRIES, backoff=BACKOFF_SECONDS, sleep=time.sleep):
        self.geocoding_url, self.forecast_url = geocoding_url, forecast_url
        self.timeout, self.max_retries, self.backoff, self.sleep = timeout, max_retries, backoff, sleep
        self.session = requests.Session()           # reuses the connection between requests
        self.session.headers["User-Agent"] = "veda-intern-api-client/1.0"
        self.raw = {}                               # last raw responses, for --raw and --save

    def close(self):
        self.session.close()

    def get_json(self, url, params):
        """GET a URL and return the JSON object, or raise an ApiError. Retries temporary failures."""
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            wait = None
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.Timeout:
                error = NetworkError("The request timed out. The server is slow or unreachable.")
            except requests.exceptions.ConnectionError:
                error = NetworkError("Could not connect to the server. Check your internet connection.")
            except requests.exceptions.RequestException as exc:
                error = NetworkError(f"The request failed ({type(exc).__name__}).")
            else:
                status = response.status_code
                if status == 200:
                    return self._read_json(response)
                if 200 < status < 300:
                    raise BadResponseError(f"Unexpected success status HTTP {status} (expected 200).")
                error = HttpStatusError(status, explain_status(status, error_reason(response)))
                if not (status == 429 or status >= 500):
                    raise error                      # permanent problem: retrying would not help
                if status == 429:
                    wait = retry_after_seconds(response)
            if attempt == attempts - 1:
                raise error                          # out of attempts
            self.sleep(wait if wait is not None else self.backoff * 2 ** attempt)

    @staticmethod
    def _read_json(response):
        try:
            data = response.json()
        except ValueError:
            raise BadResponseError("The server replied, but the response was not valid JSON.") from None
        if not isinstance(data, dict):
            raise BadResponseError("The response had an unexpected structure (expected a JSON object).")
        return data

    def find_city(self, name):
        """Geocoding: city name -> {name, admin1, country, latitude, longitude, timezone}."""
        data = self.get_json(self.geocoding_url, {"name": name, "count": 1, "language": "en", "format": "json"})
        self.raw["geocoding"] = data
        results = data.get("results")
        if not results:                              # Open-Meteo leaves "results" out when nothing matches
            raise CityNotFoundError(f"No city named '{name}' was found. Check the spelling or try a larger nearby city.")
        if not isinstance(results, list) or not isinstance(results[0], dict):
            raise BadResponseError("The location data had an unexpected structure.")
        place = results[0]
        if not (is_number(place.get("latitude")) and is_number(place.get("longitude"))):
            raise BadResponseError("The location data was incomplete (no coordinates).")
        return {key: place.get(key) for key in ("name", "admin1", "country", "latitude", "longitude", "timezone")}

    def get_weather(self, latitude, longitude):
        """Forecast: coordinates -> the full JSON response (a dict with 'current' and 'daily')."""
        data = self.get_json(self.forecast_url, {
            "latitude": latitude, "longitude": longitude, "current": CURRENT_VARIABLES,
            "daily": DAILY_VARIABLES, "timezone": "auto", "forecast_days": FORECAST_DAYS})
        self.raw["forecast"] = data
        if not isinstance(data.get("current"), dict) or not data["current"]:
            raise BadResponseError("The weather data was missing its 'current' section.")
        return data


# ---------- Formatting: turn the JSON into clean text ----------
def describe_weather(code):
    if not is_number(code):
        return "n/a"
    return WEATHER_CODES.get(int(code), f"Unknown (code {int(code)})")


def number(value, unit="", digits=1):
    """Format a number, or 'n/a' when the API sent null or something that is not a number."""
    if not is_number(value):
        return "n/a"
    return f"{value:.{digits}f}{unit}"


def format_time(text):
    try:
        return datetime.fromisoformat(text).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return "n/a"


def format_weekday(text):
    try:
        return datetime.fromisoformat(text).strftime("%a")
    except (TypeError, ValueError):
        return "---"


def format_place(place):
    return ", ".join(part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part)


def format_weather(place, data):
    """Build the weather card shown to the user."""
    current = data["current"]
    units = data.get("current_units") or {}
    temp_unit = " " + units.get("temperature_2m", "°C")
    title = f" Weather in {format_place(place)}"
    lines = ["=" * 62, title, "=" * 62,
             f" Observed at   : {format_time(current.get('time'))} ({data.get('timezone', 'local time')})",
             f" Condition     : {describe_weather(current.get('weather_code'))}",
             f" Temperature   : {number(current.get('temperature_2m'), temp_unit)}"
             f" (feels like {number(current.get('apparent_temperature'), temp_unit)})",
             f" Humidity      : {number(current.get('relative_humidity_2m'), ' ' + units.get('relative_humidity_2m', '%'), 0)}",
             f" Wind          : {number(current.get('wind_speed_10m'), ' ' + units.get('wind_speed_10m', 'km/h'))}",
             f" Precipitation : {number(current.get('precipitation'), ' ' + units.get('precipitation', 'mm'))}"]

    daily = data.get("daily")
    columns = ("time", "weather_code", "temperature_2m_min", "temperature_2m_max", "precipitation_sum")
    if isinstance(daily, dict) and all(isinstance(daily.get(c), list) for c in columns):
        rows = min(len(daily[c]) for c in columns)
        lines += ["", f" {rows}-day forecast",
                  f" {'Date':<11}{'Day':<5}{'Condition':<24}{'Low':>7}{'High':>7}{'Rain':>9}"]
        for i in range(rows):
            lines.append(f" {str(daily['time'][i]):<11}{format_weekday(daily['time'][i]):<5}"
                         f"{describe_weather(daily['weather_code'][i]):<24}"
                         f"{number(daily['temperature_2m_min'][i]):>7}{number(daily['temperature_2m_max'][i]):>7}"
                         f"{number(daily['precipitation_sum'][i], ' mm'):>9}")
    else:
        lines += ["", " (The daily forecast was not included in the response.)"]
    lines.append("=" * 62)
    return "\n".join(lines)


# ---------- Command line ----------
def validate_city(text):
    """Return (clean name, None) or (None, error message)."""
    name = " ".join(text.split())
    if not name:
        return None, "Please enter a city name."
    if len(name) > 80:
        return None, "That name is too long (maximum 80 characters)."
    if not any(ch.isalpha() for ch in name):
        return None, "A city name must contain letters."
    if any(ord(ch) < 32 for ch in name):
        return None, "The city name contains invalid characters."
    return name, None


def lookup(client, city, raw=False, save=False):
    """Fetch and print the weather for one city. Returns True on success."""
    name, problem = validate_city(city)
    if problem:
        print(f"Error: {problem}")
        return False
    try:
        place = client.find_city(name)
        data = client.get_weather(place["latitude"], place["longitude"])
        card = format_weather(place, data)
    except ApiError as error:
        print(f"Error: {error}")
        return False
    except (KeyError, TypeError, ValueError, IndexError):
        print("Error: The data from the API was in an unexpected format.")
        return False
    print(card)
    if raw:
        print("\nRaw JSON response (forecast):")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    if save:
        with open(SAVE_FILE, "w", encoding="utf-8") as file:
            json.dump(client.raw, file, indent=2, ensure_ascii=False)
        print(f"\nRaw responses saved to {os.path.basename(SAVE_FILE)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Show current weather and a 3-day forecast for a city (Open-Meteo API).")
    parser.add_argument("city", nargs="*", help="city name (leave empty for interactive mode)")
    parser.add_argument("--raw", action="store_true", help="also print the raw JSON response")
    parser.add_argument("--save", action="store_true", help="save the raw JSON responses to last_response.json")
    parser.add_argument("--timeout", type=float, default=None, help="seconds to wait for each reply (default 5 to connect, 10 to read)")
    parser.add_argument("--geocoding-url", default=GEOCODING_URL, help="geocoding endpoint (for testing or a self-hosted server)")
    parser.add_argument("--forecast-url", default=FORECAST_URL, help="forecast endpoint (for testing or a self-hosted server)")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")     # never crash on a console that cannot show a symbol
    timeout = args.timeout if args.timeout else DEFAULT_TIMEOUT
    client = ApiClient(args.geocoding_url, args.forecast_url, timeout)
    try:
        if args.city:
            return 0 if lookup(client, " ".join(args.city), args.raw, args.save) else 1
        print("Weather lookup (data from Open-Meteo, no API key needed).")
        while True:
            text = input("\nEnter a city name (or Q to quit): ")
            if text.strip().lower() in ("q", "quit", "exit"):
                print("Goodbye!")
                return 0
            lookup(client, text, args.raw, args.save)
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())