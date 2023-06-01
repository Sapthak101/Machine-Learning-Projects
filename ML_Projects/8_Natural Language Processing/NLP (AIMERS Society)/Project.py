import pyshorteners

def shorten_url(url):
    # Create an instance of the URL shortener
    shortener = pyshorteners.Shortener()

    # Shorten the given URL
    shortened_url = shortener.tinyurl.short(url)
    return shortened_url

# Example usage
long_url = "https://example.com/very/long/url"
short_url = shorten_url(long_url)
print(f"Short URL: {short_url}")
