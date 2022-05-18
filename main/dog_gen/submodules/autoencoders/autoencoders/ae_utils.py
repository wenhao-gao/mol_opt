

def repeated(f, n):
    """
    return a new function which is f composed n times
    """
    def rfun(p):
        acc = p
        for _ in range(n):
            acc = f(acc)
        return acc
    return rfun

