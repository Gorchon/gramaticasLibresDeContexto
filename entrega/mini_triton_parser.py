#!/usr/bin/env python3
"""
Mini-Triton Parser
Actividad 3.2 - Gramaticas Libres de Contexto
TC3002B - Compiladores

Punto de entrada del programa.
Uso: python3 mini_triton_parser.py <archivo.mt>

Si el archivo es valido: imprime VALIDO y el AST.
Si es invalido: imprime INVALIDO y el mensaje de error.
"""

import sys

from src.lexer import LexError, lex
from src.parser import ParseError, Parser


def main():
    if len(sys.argv) < 2:
        print(f"Uso: python3 {sys.argv[0]} <archivo_fuente>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        with open(filepath, "r") as f:
            source = f.read()
    except FileNotFoundError:
        print(f"Error: no se encontro el archivo '{filepath}'")
        sys.exit(1)

    try:
        tokens = lex(source)
        parser = Parser(tokens)
        ast = parser.parse_program()
        print("VALIDO")
        print(ast.pretty())
    except LexError as e:
        print("INVALIDO")
        print(e.message)
        print(f"En: linea {e.line}, columna {e.col}")
    except ParseError as e:
        print("INVALIDO")
        print(e.message)
        print(f"En: linea {e.line}, columna {e.col}")


if __name__ == "__main__":
    main()
