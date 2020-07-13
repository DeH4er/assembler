use super::combinators::*;

pub fn any_char(input: &str) -> ParseRes<char> {
    let mut chars = input.chars();

    match chars.next() {
        Some(ch) => Some((&input[1..], ch)),
        None => None
    }
}

pub fn char<'a>(ch: char)
    -> impl Parser<'a, char>
{
    move |input: &'a str| {
        match input.chars().next() {
            Some(input_ch) => if input_ch == ch {
                Some((&input[1..], ch))
            } else {
                None
            },
            None => None
        }
    }
}

pub fn string<'a>(s: &'a str)
    -> impl Parser<'a, &'a str>
{
    move |input: &'a str| {
        if input.len() < s.len() {
            return None;
        }

        if &input[0 .. s.len()] == s {
            Some((&input[s.len() ..], s))
        } else {
            None
        }
    }
}

pub fn string_n<'a>(n: usize)
    -> impl Parser<'a, &'a str>
{
    move |input: &'a str| {
        if input.len() < n {
            None
        } else {
            Some((&input[n ..], &input[0 .. n]))
        }
    }
}

pub fn number(input: &str) -> ParseRes<i32> {
    let mut matched = String::new();
    let mut chars = input.chars();

    match chars.next() {
        Some(ch) => {
            if ch == '-' {
                match chars.next() {
                    Some(ch2) => {
                        if ch2.is_numeric() {
                            matched.push(ch);
                            matched.push(ch2);
                        } else {
                            return None
                        }
                    },
                    None => return None
                }
            } else if ch.is_numeric() {
                matched.push(ch);
            } else {
                return None;
            }
        },
        None => {
            return None;
        }
    }

    while let Some(ch) = chars.next() {
        if ch.is_numeric() {
            matched.push(ch);
        } else {
            break;
        }
    }

    let next_index = matched.len();
    let num = matched.parse::<i32>().unwrap();

    Some((&input[next_index .. ], num))
}

pub fn space0<'a>()
    -> impl Parser<'a, ()>
{
    zero_or_more(space()).map(|_| ())
}

pub fn space1<'a>()
    -> impl Parser<'a, ()>
{
    one_or_more(space()).map(|_| ())
}

pub fn space<'a>()
    -> impl Parser<'a, char>
{
    char(' ').or(char('\n'))
}

pub fn ignore_spaces<'a, P, T>(p: P)
    -> impl Parser<'a, T>
where
    P: Parser<'a, T> + 'a,
    T: 'a
{
    space0()
        .right(p)
        .left(space0())
}
