from .cli import build_parser
from .config import Config
from .cli import run


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = Config.load(args.config)
    config.merge_cli(args)
    run(args, config)


if __name__ == "__main__":
    main()
