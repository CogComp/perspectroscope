entry_url = "https://www.googleapis.com/customsearch/v1/siterestrict"

class CustomSearchClient:
    def __init__(self, key, cx):
        """

        :param key: API key
        :param cx:
        :param q:
        """
        self._key = key
        self._cx = cx

    def query(self, q):
        pass


if __name__ == '__main__':
    from sys import argv
    if len(argv) != 4:
        print("Usage: python google_custom_search.py [api-key] [cx] [query]")
        exit(1)
    cli = CustomSearchClient(argv[1], argv[2])
    print(cli.query(argv[3]))