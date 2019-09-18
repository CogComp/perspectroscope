from googleapiclient.discovery import build

entry_url = "https://www.googleapis.com/customsearch/v1/siterestrict"

class CustomSearchClient:
    def __init__(self, key, cx):
        """

        :param key: API key
        :param cx: custom search engine id
        :param q: query
        """
        self._cx = cx

        self._service = build("customsearch", "v1", developerKey=key)

    def query(self, q, **kwargs):
        res = self._service.cse().list(q=q, cx=self._cx, **kwargs).execute()
        if 'items' in res:
            return res['items']
        else:
            return []


if __name__ == '__main__':
    from sys import argv
    if len(argv) != 4:
        print("Usage: python google_custom_search.py [api-key] [cx] [query]")
        exit(1)
    cli = CustomSearchClient(argv[1], argv[2])
    r = cli.query(argv[3])
    print(len(r))
    print([c["link"] for c in r])
