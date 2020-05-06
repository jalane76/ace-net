# -*- coding: utf-8 -*-
import click

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    

if __name__ == '__main__':
    main()
