pub type ParseRes<'a, Output> = Option<(&'a str, Output)>;

pub trait Parser<'a, Output> {
    fn parse(&self, input: &'a str) -> ParseRes<'a, Output>;

    fn map<F, NewOutput>(self, map_fn: F)
        -> BoxedParser<'a, NewOutput>
    where
        Self: Sized + 'a,
        Output: 'a,
        NewOutput: 'a,
        F: Fn(Output) -> NewOutput + 'a
    {
        BoxedParser::new(map(self, map_fn))
    }

    fn pair<P, NewOutput>(self, p: P)
        -> BoxedParser<'a, (Output, NewOutput)>
    where
        P: Parser<'a, NewOutput> + 'a,
        Self: Sized + 'a,
        Output: 'a,
        NewOutput: 'a,
    {
        BoxedParser::new(pair(self, p))
    }

    fn left<P, NewOutput>(self, p: P)
        -> BoxedParser<'a, Output>
    where
        P: Parser<'a, NewOutput> + 'a,
        Self: Sized + 'a,
        Output: 'a,
        NewOutput: 'a,
    {
        BoxedParser::new(left(self, p))
    }

    fn pred<F>(self, f: F)
        -> BoxedParser<'a, Output>
    where
        F: Fn(&Output) -> bool,
        F: 'a,
        Self: Sized + 'a,
        Output: 'a,
    {
        BoxedParser::new(pred(self, f))
    }

    fn right<P, NewOutput>(self, p: P)
        -> BoxedParser<'a, NewOutput>
    where
        P: Parser<'a, NewOutput> + 'a,
        Self: Sized + 'a,
        Output: 'a,
        NewOutput: 'a,
    {
        BoxedParser::new(right(self, p))
    }

    fn or<P>(self, parser: P)
        -> BoxedParser<'a, Output>
    where
        Self: Sized + 'a,
        Output: 'a,
        P: 'a,
        P: Parser<'a, Output>
    {
        BoxedParser::new(or(self, parser))
    }

    fn debug(self, name: &'a str)
        -> BoxedParser<'a, Output>
    where
        Self: Sized + 'a,
        Output: std::fmt::Debug + 'a
    {
        BoxedParser::new(debug(name, self))
    }
}

impl<'a, F, Output> Parser<'a, Output> for F
where
    F: Fn(&'a str) -> ParseRes<Output>,
{
    fn parse(&self, input: &'a str) -> ParseRes<'a, Output> {
        self(input)
    }
}

pub struct BoxedParser<'a, Output> {
    parser: Box<dyn Parser<'a, Output> + 'a>,
}

impl<'a, Output> BoxedParser<'a, Output> {
    pub fn new<P>(parser: P) -> Self
    where
        P: Parser<'a, Output> + 'a
    {
        BoxedParser {
            parser: Box::new(parser)
        }
    }
}

impl<'a, Output> Parser<'a, Output> for BoxedParser<'a, Output> {
    fn parse(&self, input: &'a str) -> ParseRes<'a, Output> {
        self.parser.parse(input)
    }
}

pub struct LazyParser<'a, Output> {
    constructor: Box<dyn Fn() -> BoxedParser<'a, Output> + 'a>
}

impl<'a, Output> LazyParser<'a, Output> {
    pub fn new<F>(constructor: F) -> Self
    where
        F: Fn() -> BoxedParser<'a, Output> + 'a
    {
        LazyParser { constructor: Box::new(constructor) }
    }
}

impl<'a, Output> Parser<'a, Output> for LazyParser<'a, Output> {
    fn parse(&self, input: &'a str) -> ParseRes<'a, Output> {
        (self.constructor)().parse(input)
    }
}

pub fn map<'a, P, F, A, B>(parser: P, map_fn: F)
    -> impl Parser<'a, B>
where
    P: Parser<'a, A>,
    F: Fn(A) -> B
{
    move |input|
        parser.parse(input)
            .map(|(next_input, res)| (next_input, map_fn(res)))
}

fn pred<'a, P, F, T>(p: P, f: F)
    -> impl Parser<'a, T>
where
    P: Parser<'a, T>,
    F: Fn(&T) -> bool
{
    move |input: &'a str| {
        if let Some((next_input, res)) = p.parse(input) {
            if f(&res) {
                return Some((next_input, res));
            }
        }

        None
    }
}

fn and1<'a, P1, P2, R1, R2>(p1: P1, p2: P2)
    -> impl Parser<'a, (R1, R2)>
where
    P1: Parser<'a, R1>,
    P2: Parser<'a, R2>,
{
    move |input| {
        if let Some((next_input, res1)) = p1.parse(input) {
            if let Some((_, res2)) = p2.parse(input) {
                return Some((next_input, (res1, res2)));
            }
        }

        None
    }
}

fn debug<'a, P, T>(name: &'a str, p: P)
    -> impl Parser<'a, T>
where
    P: Parser<'a, T>,
    T: std::fmt::Debug
{
    move |input| {
        let res = p.parse(input);
        println!("======================================================");

        if res.is_none() {
            println!("Parser {} failed:\n{:?}", name, input);
        } else {
            println!("Parser {} ok:\n{:?}", name, input);
            println!("Parser {} result:\n{:?}", name, res.as_ref().unwrap());
        }

        println!();
        res
    }
}

pub fn pair<'a, P1, P2, R1, R2>(parser1: P1, parser2: P2)
    -> impl Parser<'a, (R1, R2)>
where
    P1: Parser<'a, R1>,
    P2: Parser<'a, R2>
{
    move |input| match parser1.parse(input) {
        Some((next_input, result1)) => match parser2.parse(next_input) {
            Some((final_input, result2)) => Some((final_input, (result1, result2))),
            None => None,
        },
        None => None,
    }
}

pub fn left<'a, P1, P2, R1, R2>(parser1: P1, parser2: P2) -> impl Parser<'a, R1>
where
    P1: Parser<'a, R1> + 'a,
    P2: Parser<'a, R2> + 'a,
    R1: 'a,
    R2: 'a
{
    parser1
        .pair(parser2)
        .map(|(left, _)| left)
}

pub fn right<'a, P1, P2, R1, R2>(parser1: P1, parser2: P2) -> impl Parser<'a, R2>
where
    P1: Parser<'a, R1> + 'a,
    P2: Parser<'a, R2> + 'a,
    R1: 'a,
    R2: 'a
{
    parser1
        .pair(parser2)
        .map(|(_, right)| right)
}

pub fn one_or_more<'a, P, T>(parser: P)
    -> impl Parser<'a, Vec<T>>
where
    P: Parser<'a, T>
{
    move |mut input| {
        let mut res = vec![];

        if let Some((next_input, val)) = parser.parse(input) {
            input = next_input;
            res.push(val);
        } else {
            return None;
        }

        while let Some((next_input, val)) = parser.parse(input) {
            input = next_input;
            res.push(val);
        }

        Some((input, res))
    }
}

pub fn zero_or_more<'a, P, T>(parser: P)
    -> impl Parser<'a, Vec<T>>
where
    P: Parser<'a, T>
{
    move |mut input| {
        let mut res = vec![];

        while let Some((next_input, val)) = parser.parse(input) {
            input = next_input;
            res.push(val);
        }

        Some((input, res))
    }
}

pub fn or<'a, P1, P2, T>(parser1: P1, parser2: P2)
    -> impl Parser<'a, T>
where
    P1: Parser<'a, T>,
    P2: Parser<'a, T>,
{
    move |input| match parser1.parse(input) {
        some @ Some(_) => some,
        None => parser2.parse(input)
    }
}

pub fn lazy<'a, P, T, F>(f: F)
    -> LazyParser<'a, T>
where
    P: Parser<'a, T> + 'a,
    F: Fn() -> P + 'a,
{
    LazyParser::new(move || BoxedParser::new(f()))
}

