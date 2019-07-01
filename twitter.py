import tweepy

class Twitter :
    def __init__(self):
        print("hehe")
        pass

    def instance(self):
        CONSUMER_KEY = "ARhmOYK2f5xR5PGCtUPZwNayu"
        CONSUMER_SECRET = "Pt6Z55l8RynOvEcLyYtxR1uarOKhar939YnsVI0S7OcLZjGPtE"
        ACCESS_KEY = "1031914262896074759-UKdpwFlj8Ylw3zLXjINAjIEwb2VoSi"
        ACCESS_SECRET = "JYJenkiJP6UEzoQljCikqWb0cImGoyTG6uz8xZUpDpXD1"
        api = tweepy.OAuthHandler(consumer_key = CONSUMER_KEY, consumer_secret = CONSUMER_SECRET)
        api.set_access_token(ACCESS_KEY, ACCESS_SECRET)
        return tweepy.API(api, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
