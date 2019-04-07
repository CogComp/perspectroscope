from newspaper import Article

def parse_article(url, language='en'):
    a = Article(url, language=language)
    a.download()
    a.parse()

    return a

if __name__ == '__main__':
    a = parse_article('https://www.nytimes.com/2014/11/18/upshot/got-milk-might-not-be-doing-you-much-good.html')
    # print(a.title)
    # print(a.text)
    print([p for p in a.text.splitlines() if p])