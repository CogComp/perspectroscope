from newspaper import Article, Config

config = Config()
config.memoize_articles = False
config.language = 'en'


def parse_article(url, language='en'):
    """
    TODO: support other languages
    :param url:
    :param language:
    :return:
    """
    a = Article(url, config)
    a.download()
    a.parse()

    return a

if __name__ == '__main__':
    a = parse_article('https://www.nytimes.com/2014/11/18/upshot/got-milk-might-not-be-doing-you-much-good.html')
    # print(a.title)
    # print(a.text)
    print([p for p in a.text.splitlines() if p])