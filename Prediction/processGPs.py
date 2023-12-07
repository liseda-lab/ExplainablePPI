import sympy

def processGPmodel(expression, aspects):
    locals = {
        "add": sympy.Add,
        "mul": sympy.Mul,
        "sub": lambda x, y: x - y,
        "div": lambda x, y: x / y,
        "max": sympy.Max,
        "min": sympy.Min}

    for i in range(len(aspects)):
        X = sympy.symbols(aspects[i])
        locals["X" + str(i)] = X

    expression_simplified = sympy.sympify(expression, locals=locals)
    expression_simplified = sympy.simplify(expression_simplified)

    return expression_simplified