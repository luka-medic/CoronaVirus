

def generate_lms_function(distribution, mean=None, median=None, std=None, p5=None, p95=None, p99=None, loc=True):
    def function2(x):
        funcs = []
        s,scale=x[:2]
        loc = 0
        if mean is not None:
            funcs.append(lambda s,loc,scale: (distribution.mean(s,loc,scale) - mean) ** 2)
        if median is not None:
            funcs.append(lambda s,loc,scale: (distribution.median(s,loc,scale) - median) ** 2)
        if std is not None:
            funcs.append(lambda s,loc,scale: (distribution.std(s,loc,scale) - std) ** 2)
        if p5 is not None:
            funcs.append(lambda s,loc,scale: (distribution.cdf(p5, s,loc,scale) - 0.05) ** 2)
        if p95 is not None:
            funcs.append(lambda s,loc,scale: (distribution.cdf(p95, s,loc,scale) - 0.95) ** 2)
        if p99 is not None:
            funcs.append(lambda s,loc,scale: (distribution.cdf(p99, s,loc,scale) - 0.99) ** 2)
        return sum(f(s,loc,scale) for f in funcs)

    def function(x):
        funcs = []
        if mean is not None:
            funcs.append(lambda *x: (distribution.mean(*x) - mean) ** 2)
        if median is not None:
            funcs.append(lambda *x: (distribution.median(*x) - median) ** 2)
        if std is not None:
            funcs.append(lambda *x: (distribution.std(*x) - std) ** 2)
        if p5 is not None:
            funcs.append(lambda *x: (distribution.cdf(p5, *x) - 0.05) ** 2)
        if p95 is not None:
            funcs.append(lambda *x: (distribution.cdf(p95, *x) - 0.95) ** 2)
        if p99 is not None:
            funcs.append(lambda *x: (distribution.cdf(p99, *x) - 0.99) ** 2)
        return sum(f(*x) for f in funcs)

    if loc:
        return function
    else:
        return function2