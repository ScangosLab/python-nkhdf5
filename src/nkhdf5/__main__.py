"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """NKHDF5."""


if __name__ == "__main__":
    main(prog_name="python-nkhdf5")  # pragma: no cover
