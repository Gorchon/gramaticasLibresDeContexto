# Mini-Triton Parser

## Actividad 3.2 — Gramáticas Libres de Contexto

Parser de descenso recursivo para el lenguaje Mini-Triton, implementado en Python 3.

### Ejecución

```bash
cd entrega/
python3 mini_triton_parser.py <archivo.mt>
```

### Ejemplos

```bash
python3 mini_triton_parser.py tests/01_valid_simple_add.mt    # VALIDO + AST
python3 mini_triton_parser.py tests/08_invalid_no_decorator.mt # INVALIDO + error
```

### Estructura

```
entrega/
├── mini_triton_parser.py    <- Punto de entrada
├── src/
│   ├── lexer.py             <- Analizador léxico
│   ├── parser.py            <- Parser (descenso recursivo LL)
│   └── ast_nodes.py         <- Nodos del AST
├── tests/                   <- 13 casos de prueba
└── reporte.md               <- Documento con gramática, diseño y resultados
```

### Requisitos

Python 3.6+. Sin dependencias externas.
