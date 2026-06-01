"""
Mini-Triton: Nodos del Arbol Sintactico Abstracto (AST)
Actividad 3.2 - Gramaticas Libres de Contexto
TC3002B - Compiladores

Define los tipos de nodo que componen el AST.
Cada nodo tiene un metodo pretty() para impresion legible.
"""


class ASTNode:
    """Clase base para todos los nodos del AST."""

    def pretty(self, indent=0):
        raise NotImplementedError


class Program(ASTNode):
    """Nodo raiz: contiene un unico kernel."""

    def __init__(self, kernel):
        self.kernel = kernel

    def pretty(self, indent=0):
        return self.kernel.pretty(indent)


class Kernel(ASTNode):
    """Definicion de un kernel: nombre, parametros y cuerpo."""

    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

    def pretty(self, indent=0):
        pad = "  " * indent
        params_str = ", ".join(p.pretty() for p in self.params)
        lines = [f"{pad}Kernel(name=\"{self.name}\", params=[{params_str}])"]
        lines.append(f"{pad}  body:")
        for stmt in self.body:
            lines.append(stmt.pretty(indent + 2))
        return "\n".join(lines)


class Param(ASTNode):
    """Parametro de funcion, opcionalmente con anotacion de tipo."""

    def __init__(self, name, annotation=None):
        self.name = name
        self.annotation = annotation

    def pretty(self, indent=0):
        if self.annotation:
            return f"{self.name}: {self.annotation}"
        return self.name


class Assign(ASTNode):
    """Sentencia de asignacion: id = expr"""

    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

    def pretty(self, indent=0):
        pad = "  " * indent
        return f"{pad}Assign({self.name}, {self.expr.pretty()})"


class ExprStmt(ASTNode):
    """Sentencia de expresion (una expresion sola como sentencia)."""

    def __init__(self, expr):
        self.expr = expr

    def pretty(self, indent=0):
        pad = "  " * indent
        return f"{pad}ExprStmt({self.expr.pretty()})"


class BinaryOp(ASTNode):
    """Operacion binaria: left op right"""

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def pretty(self, indent=0):
        return f"BinaryOp(\"{self.op}\", {self.left.pretty()}, {self.right.pretty()})"


class Call(ASTNode):
    """Llamada a funcion: nombre(args)"""

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def pretty(self, indent=0):
        args_str = ", ".join(a.pretty() for a in self.args)
        return f"Call(\"{self.name}\", [{args_str}])"


class Name(ASTNode):
    """Referencia a una variable o identificador."""

    def __init__(self, value):
        self.value = value

    def pretty(self, indent=0):
        return f"Name({self.value})"


class Number(ASTNode):
    """Literal numerico."""

    def __init__(self, value):
        self.value = value

    def pretty(self, indent=0):
        return f"Number({self.value})"
