import click

@click.command("bored")

def bored (x : int) -> float:
    return 15/x

print(bored ())

