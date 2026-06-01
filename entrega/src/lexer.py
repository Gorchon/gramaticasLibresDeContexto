"""
Mini-Triton: Analizador Lexico (Lexer)
Actividad 3.2 - Gramaticas Libres de Contexto
TC3002B - Compiladores

Tokeniza un archivo fuente Mini-Triton en una secuencia de tokens.
Cada token tiene: tipo, lexema, valor y posicion (linea, columna).
"""


class Token:
    """Representa un token individual del codigo fuente."""

    def __init__(self, type, lexeme, value, line, col):
        self.type = type
        self.lexeme = lexeme
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type}, {self.lexeme!r}, linea={self.line})"


# Palabras reservadas del lenguaje Mini-Triton
KEYWORDS = {"def", "tl"}

# Tipos de token
TT_AT       = "AT"          # @
TT_DOT      = "DOT"         # .
TT_COMMA    = "COMMA"       # ,
TT_COLON    = "COLON"       # :
TT_SEMI     = "SEMICOLON"   # ;
TT_ASSIGN   = "ASSIGN"      # =
TT_PLUS     = "PLUS"        # +
TT_MINUS    = "MINUS"       # -
TT_STAR     = "STAR"        # *
TT_SLASH    = "SLASH"       # /
TT_LPAREN   = "LPAREN"      # (
TT_RPAREN   = "RPAREN"      # )
TT_LBRACE   = "LBRACE"      # {
TT_RBRACE   = "RBRACE"      # }
TT_ID       = "ID"          # identificador
TT_NUMBER   = "NUMBER"      # literal numerico
TT_DEF      = "DEF"         # palabra reservada def
TT_EOF      = "EOF"         # fin de archivo


class LexError(Exception):
    """Error durante el analisis lexico."""

    def __init__(self, message, line, col):
        self.message = message
        self.line = line
        self.col = col
        super().__init__(f"Error lexico en linea {line}, columna {col}: {message}")


def lex(source):
    """
    Recibe el codigo fuente como string.
    Retorna una lista de Token.
    Lanza LexError si encuentra un caracter invalido.
    """
    tokens = []
    i = 0
    line = 1
    col = 1
    n = len(source)

    while i < n:
        ch = source[i]

        # Saltos de linea
        if ch == "\n":
            line += 1
            col = 1
            i += 1
            continue

        # Espacios y tabs
        if ch in " \t\r":
            col += 1
            i += 1
            continue

        # Comentarios de linea (estilo Python: #)
        if ch == "#":
            while i < n and source[i] != "\n":
                i += 1
            continue

        # Simbolos de un caracter
        simple = {
            "@": TT_AT, ".": TT_DOT, ",": TT_COMMA, ":": TT_COLON,
            ";": TT_SEMI, "=": TT_ASSIGN, "+": TT_PLUS, "-": TT_MINUS,
            "*": TT_STAR, "/": TT_SLASH, "(": TT_LPAREN, ")": TT_RPAREN,
            "{": TT_LBRACE, "}": TT_RBRACE,
        }

        if ch in simple:
            tokens.append(Token(simple[ch], ch, None, line, col))
            i += 1
            col += 1
            continue

        # Numeros (enteros y flotantes)
        if ch.isdigit():
            start = i
            start_col = col
            while i < n and source[i].isdigit():
                i += 1
                col += 1
            # Parte decimal opcional
            if i < n and source[i] == "." and i + 1 < n and source[i + 1].isdigit():
                i += 1
                col += 1
                while i < n and source[i].isdigit():
                    i += 1
                    col += 1
            lexeme = source[start:i]
            value = float(lexeme) if "." in lexeme else int(lexeme)
            tokens.append(Token(TT_NUMBER, lexeme, value, line, start_col))
            continue

        # Identificadores y palabras reservadas
        if ch.isalpha() or ch == "_":
            start = i
            start_col = col
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
                col += 1
            lexeme = source[start:i]
            if lexeme == "def":
                tokens.append(Token(TT_DEF, lexeme, None, line, start_col))
            else:
                tokens.append(Token(TT_ID, lexeme, lexeme, line, start_col))
            continue

        # Caracter no reconocido
        raise LexError(f"caracter inesperado '{ch}'", line, col)

    tokens.append(Token(TT_EOF, "", None, line, col))
    return tokens
