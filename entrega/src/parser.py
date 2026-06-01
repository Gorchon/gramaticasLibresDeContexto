"""
Mini-Triton: Parser (Analizador Sintactico)
Actividad 3.2 - Gramaticas Libres de Contexto
TC3002B - Compiladores

Implementa un parser de descenso recursivo (recursive descent, LL(1))
para el lenguaje Mini-Triton. Construye un AST a partir de la
secuencia de tokens producida por el lexer.

Gramatica implementada (EBNF):

  Program     -> Decorator Kernel EOF
  Decorator   -> '@' 'triton' '.' 'jit'
  Kernel      -> 'def' ID '(' ParamList ')' ':' '{' StmtList '}'
  ParamList   -> Param (',' Param)* | epsilon
  Param       -> ID (':' QualifiedID)?
  StmtList    -> Stmt*
  Stmt        -> Assign | ExprStmt
                 (si lookahead es ID y peek(1) es '=', es Assign)
  Assign      -> ID '=' Expr ';'
  ExprStmt    -> Expr ';'
  Expr        -> Term (('+' | '-') Term)*
  Term        -> Factor (('*' | '/') Factor)*
  Factor      -> NUMBER | '(' Expr ')' | NameOrCall
  NameOrCall  -> QualifiedID ( '(' ArgList ')' )?
  QualifiedID -> ID ('.' ID)*
  ArgList     -> Expr (',' Expr)* | epsilon
"""

from .lexer import (
    Token, TT_AT, TT_DOT, TT_COMMA, TT_COLON, TT_SEMI,
    TT_ASSIGN, TT_PLUS, TT_MINUS, TT_STAR, TT_SLASH,
    TT_LPAREN, TT_RPAREN, TT_LBRACE, TT_RBRACE,
    TT_ID, TT_NUMBER, TT_DEF, TT_EOF,
)
from .ast_nodes import (
    Program, Kernel, Param, Assign, ExprStmt,
    BinaryOp, Call, Name, Number,
)


class ParseError(Exception):
    """Error durante el analisis sintactico."""

    def __init__(self, message, token):
        self.message = message
        self.token = token
        self.line = token.line
        self.col = token.col
        super().__init__(
            f"Error sintactico en linea {token.line}, columna {token.col}: {message}"
        )


class Parser:
    """
    Parser de descenso recursivo para Mini-Triton.
    Consume tokens uno a uno y construye el AST.
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0
        self.lookahead = tokens[0]

    # ── Utilidades ──────────────────────────────────────────────

    def advance(self):
        """Avanza al siguiente token."""
        self.i += 1
        if self.i < len(self.tokens):
            self.lookahead = self.tokens[self.i]
        return self.tokens[self.i - 1]

    def match(self, expected_type):
        """
        Consume el token actual si su tipo coincide con expected_type.
        Si no coincide, lanza un ParseError descriptivo.
        """
        if self.lookahead.type == expected_type:
            return self.advance()
        raise ParseError(
            f"se esperaba {expected_type} pero se encontro "
            f"{self.lookahead.type} ('{self.lookahead.lexeme}')",
            self.lookahead,
        )

    def peek_type(self, k=0):
        """Retorna el tipo del token en posicion i+k sin consumirlo."""
        pos = self.i + k
        if pos < len(self.tokens):
            return self.tokens[pos].type
        return TT_EOF

    def check(self, token_type):
        """Verifica si el token actual es del tipo dado, sin consumirlo."""
        return self.lookahead.type == token_type

    # ── Reglas de la gramatica ──────────────────────────────────

    def parse_program(self):
        """
        Program -> Decorator Kernel EOF
        Un programa Mini-Triton tiene exactamente un decorador y un kernel.
        """
        self.parse_decorator()
        kernel = self.parse_kernel()
        self.match(TT_EOF)
        return Program(kernel)

    def parse_decorator(self):
        """
        Decorator -> '@' 'triton' '.' 'jit'
        Verifica la secuencia exacta del decorador @triton.jit
        """
        self.match(TT_AT)
        tok = self.match(TT_ID)
        if tok.lexeme != "triton":
            raise ParseError(
                f"se esperaba 'triton' despues de '@', se encontro '{tok.lexeme}'",
                tok,
            )
        self.match(TT_DOT)
        tok = self.match(TT_ID)
        if tok.lexeme != "jit":
            raise ParseError(
                f"se esperaba 'jit' despues de 'triton.', se encontro '{tok.lexeme}'",
                tok,
            )

    def parse_kernel(self):
        """
        Kernel -> 'def' ID '(' ParamList ')' ':' '{' StmtList '}'
        Parsea la definicion completa de un kernel.
        """
        self.match(TT_DEF)
        name_tok = self.match(TT_ID)
        self.match(TT_LPAREN)
        params = self.parse_param_list()
        self.match(TT_RPAREN)
        self.match(TT_COLON)
        self.match(TT_LBRACE)
        body = self.parse_stmt_list()
        self.match(TT_RBRACE)
        return Kernel(name_tok.lexeme, params, body)

    def parse_param_list(self):
        """
        ParamList -> Param (',' Param)* | epsilon
        Lista de parametros separados por comas. Puede estar vacia.
        """
        params = []
        if self.check(TT_RPAREN):
            return params
        params.append(self.parse_param())
        while self.check(TT_COMMA):
            self.advance()
            params.append(self.parse_param())
        return params

    def parse_param(self):
        """
        Param -> ID (':' QualifiedID)?
        Un parametro puede tener anotacion opcional, por ejemplo: BS : tl.constexpr
        """
        name_tok = self.match(TT_ID)
        annotation = None
        if self.check(TT_COLON):
            self.advance()
            annotation = self.parse_qualified_id()
        return Param(name_tok.lexeme, annotation)

    def parse_stmt_list(self):
        """
        StmtList -> Stmt*
        Lista de sentencias dentro de un bloque { }.
        Termina cuando encuentra '}'.
        """
        stmts = []
        while not self.check(TT_RBRACE) and not self.check(TT_EOF):
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self):
        """
        Stmt -> Assign | ExprStmt
        Distingue entre asignacion y sentencia de expresion:
        Si el token actual es ID y el siguiente es '=', es asignacion.
        En cualquier otro caso, es sentencia de expresion.
        """
        if self.check(TT_ID) and self.peek_type(1) == TT_ASSIGN:
            return self.parse_assign()
        return self.parse_expr_stmt()

    def parse_assign(self):
        """
        Assign -> ID '=' Expr ';'
        Sentencia de asignacion simple.
        """
        name_tok = self.match(TT_ID)
        self.match(TT_ASSIGN)
        expr = self.parse_expr()
        self.match(TT_SEMI)
        return Assign(name_tok.lexeme, expr)

    def parse_expr_stmt(self):
        """
        ExprStmt -> Expr ';'
        Una expresion usada como sentencia (por ejemplo, una llamada a funcion).
        """
        expr = self.parse_expr()
        self.match(TT_SEMI)
        return ExprStmt(expr)

    def parse_expr(self):
        """
        Expr -> Term (('+' | '-') Term)*
        Expresion con operadores de suma y resta (menor precedencia).
        Asociatividad por la izquierda gracias a la iteracion.
        """
        left = self.parse_term()
        while self.check(TT_PLUS) or self.check(TT_MINUS):
            op_tok = self.advance()
            right = self.parse_term()
            left = BinaryOp(op_tok.lexeme, left, right)
        return left

    def parse_term(self):
        """
        Term -> Factor (('*' | '/') Factor)*
        Termino con operadores de multiplicacion y division (mayor precedencia que +/-).
        Asociatividad por la izquierda.
        """
        left = self.parse_factor()
        while self.check(TT_STAR) or self.check(TT_SLASH):
            op_tok = self.advance()
            right = self.parse_factor()
            left = BinaryOp(op_tok.lexeme, left, right)
        return left

    def parse_factor(self):
        """
        Factor -> NUMBER | '(' Expr ')' | NameOrCall
        Factor: la unidad de mayor precedencia.
        Puede ser un numero, una expresion entre parentesis, o un nombre/llamada.
        """
        # Literal numerico
        if self.check(TT_NUMBER):
            tok = self.advance()
            return Number(tok.value)

        # Expresion entre parentesis
        if self.check(TT_LPAREN):
            self.advance()
            expr = self.parse_expr()
            self.match(TT_RPAREN)
            return expr

        # Nombre o llamada a funcion
        if self.check(TT_ID):
            return self.parse_name_or_call()

        # Ningun patron valido
        raise ParseError(
            f"se esperaba una expresion (numero, identificador o '('), "
            f"pero se encontro {self.lookahead.type} ('{self.lookahead.lexeme}')",
            self.lookahead,
        )

    def parse_name_or_call(self):
        """
        NameOrCall -> QualifiedID ( '(' ArgList ')' )?
        Un identificador posiblemente cualificado (con puntos) que puede
        ser seguido de argumentos para formar una llamada a funcion.
        Ejemplo: tl.load(x + offs)
        """
        qualified = self.parse_qualified_id()

        # Si sigue un parentesis, es llamada a funcion
        if self.check(TT_LPAREN):
            self.advance()
            args = self.parse_arg_list()
            self.match(TT_RPAREN)
            return Call(qualified, args)

        # Si no, es solo un nombre
        return Name(qualified)

    def parse_qualified_id(self):
        """
        QualifiedID -> ID ('.' ID)*
        Permite nombres con puntos como tl.load, tl.program_id, tl.constexpr.
        Retorna un string como "tl.load".
        """
        tok = self.match(TT_ID)
        name = tok.lexeme
        while self.check(TT_DOT):
            self.advance()
            tok2 = self.match(TT_ID)
            name += "." + tok2.lexeme
        return name

    def parse_arg_list(self):
        """
        ArgList -> Expr (',' Expr)* | epsilon
        Lista de argumentos posicionales separados por comas.
        Puede estar vacia.
        """
        args = []
        if self.check(TT_RPAREN):
            return args
        args.append(self.parse_expr())
        while self.check(TT_COMMA):
            self.advance()
            args.append(self.parse_expr())
        return args
