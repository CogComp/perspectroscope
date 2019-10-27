import argparse
import json
import urllib.parse


def make_url(base: str, sandbox: bool) -> str:
    query_args_dict = {}
    if sandbox:
        query_args_dict["sandbox"] = 1

    return base + "?" + urllib.parse.urlencode(query_args_dict)


def main():
    parser = argparse.ArgumentParser(description='AMTI data builder for external URLs')
    parser.add_argument('--base', required=True, help='base url')
    parser.add_argument('--size', required=True, help='the number of HITS')
    parser.add_argument('--sandbox', action='store_true', help='use sandbox')

    args = parser.parse_args()

    for _ in range(int(args.size)):
        print(json.dumps(
            {
                "external_url": make_url(args.base, args.sandbox)
            },
            separators=(',', ':')
        ))


if __name__ == '__main__':
    main()
