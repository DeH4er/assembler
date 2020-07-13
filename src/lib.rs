use std::collections::HashMap;
use std::cmp::Ordering;

type Identifier = String;

#[derive (Clone, Debug)]
pub enum MsgItem {
    Reg(char),
    Str(String)
}

#[derive (Clone, Debug)]
pub struct Reg {
    pub key: char
}

#[derive (Clone, Debug)]
pub enum Eval {
    Reg(char),
    Num(i32)
}

#[derive (Clone, Debug)]
pub enum Instruction {
    Mov(Reg, Eval),
    Inc(Reg),
    Dec(Reg),
    Add(Reg, Eval),
    Sub(Reg, Eval),
    Mul(Reg, Eval),
    Div(Reg, Eval),
    Label(Identifier),
    Cmp(Eval, Eval),
    Jmp(Identifier),
    Jne(Identifier),
    Je(Identifier),
    Jge(Identifier),
    Jg(Identifier),
    Jle(Identifier),
    Jl(Identifier),
    Call(Identifier),
    Ret,
    Msg(Vec<MsgItem>),
    End,
    Nop
}

pub struct AssemblerInterpreter {
}

impl AssemblerInterpreter {
    pub fn interpret(input: &str) -> Option<String> {
        let instructions = parser::parse(input);

        println!();
        println!("{:?}", instructions);

        let mut interpreter = Interpreter::new(instructions);
        interpreter.interpret();
        interpreter.get_output()
    }
}

pub struct Interpreter {
    registers: HashMap<char, i32>,
    labels: HashMap<String, usize>,
    program_pointer: usize,
    instructions: Vec<Instruction>,
    output: String,
    ord: Ordering,
    call_stack: Vec<usize>,
    end: bool
}

impl Interpreter {
    pub fn new(instructions: Vec<Instruction>) -> Interpreter {
        Interpreter {
            registers: HashMap::new(),
            labels: HashMap::new(),
            program_pointer: 0,
            output: String::new(),
            instructions,
            ord: Ordering::Equal,
            call_stack: vec![],
            end: false
        }
    }

    pub fn get_output(&self) -> Option<String> {
        if self.end {
            Some(self.output.to_string())
        } else {
            None
        }
    }

    pub fn interpret(&mut self) {
        self.check_labels();
        while self.program_pointer < self.instructions.len() && !self.end {
            let instruction = self.instructions[self.program_pointer].clone();
            self.run_instruction(&instruction);
        }
    }

    fn run_instruction(&mut self, instruction: &Instruction) {
        use Instruction::*;

        match instruction {
            Mov(reg, eval) => self.run_mov(reg, eval),
            Inc(reg) => self.run_inc(reg),
            Dec(reg) => self.run_dec(reg),
            Add(reg, eval) => self.run_add(reg, eval),
            Sub(reg, eval) => self.run_sub(reg, eval),
            Mul(reg, eval) => self.run_mul(reg, eval),
            Div(reg, eval) => self.run_div(reg, eval),
            Cmp(eval1, eval2) => self.run_cmp(eval1, eval2),
            Jmp(id) => self.run_jmp(id),
            Jne(id) => self.run_jne(id),
            Je(id) => self.run_je(id),
            Jge(id) => self.run_jge(id),
            Jg(id) => self.run_jg(id),
            Jle(id) => self.run_jle(id),
            Jl(id) => self.run_jl(id),
            Call(id) => self.run_call(id),
            Ret => self.run_ret(),
            Msg(items) => self.run_msg(items),
            End => self.run_end(),
            Label(_) => self.inc_pointer(),
            Nop => self.inc_pointer(),
        }
    }

    fn run_mov(&mut self, reg: &Reg, eval: &Eval) {
        self.modify_reg_by_eval(reg, eval, |_, e| e);
        self.inc_pointer();
    }

    fn run_inc(&mut self, reg: &Reg) {
        self.modify_reg(reg, |v| v + 1);
        self.inc_pointer();
    }

    fn run_dec(&mut self, reg: &Reg) {
        self.modify_reg(reg, |v| v - 1);
        self.inc_pointer();
    }

    fn run_add(&mut self, reg: &Reg, eval: &Eval) {
        self.modify_reg_by_eval(reg, eval, |v, e| v + e);
        self.inc_pointer();
    }

    fn run_sub(&mut self, reg: &Reg, eval: &Eval) {
        self.modify_reg_by_eval(reg, eval, |v, e| v - e);
        self.inc_pointer();
    }

    fn run_mul(&mut self, reg: &Reg, eval: &Eval) {
        self.modify_reg_by_eval(reg, eval, |v, e| v * e);
        self.inc_pointer();
    }

    fn run_div(&mut self, reg: &Reg, eval: &Eval) {
        self.modify_reg_by_eval(reg, eval, |v, e| v / e);
        self.inc_pointer();
    }

    fn run_cmp(&mut self, eval1: &Eval, eval2: &Eval) {
        let e1 = self.eval_value(eval1);
        let e2 = self.eval_value(eval2);
        self.ord = e1.cmp(&e2);
        self.inc_pointer();
    }

    fn run_jmp(&mut self, id: &Identifier) {
        self.program_pointer = *self.labels.get(id).unwrap();
    }

    fn run_jne(&mut self, id: &Identifier) {
        if self.ord == Ordering::Less || self.ord == Ordering::Greater {
            self.run_jmp(id);
        } else {
            self.inc_pointer();
        }
    }

    fn run_je(&mut self, id: &Identifier) {
        if self.ord == Ordering::Equal{
            self.run_jmp(id);
        } else {
            self.inc_pointer();
        }
    }

    fn run_jge(&mut self, id: &Identifier) {
        if self.ord == Ordering::Greater || self.ord == Ordering::Equal {
            self.run_jmp(id);
        } else {
            self.inc_pointer();
        }
    }

    fn run_jg(&mut self, id: &Identifier) {
        if self.ord == Ordering::Greater {
            self.run_jmp(id);
        } else {
            self.inc_pointer();
        }
    }

    fn run_jle(&mut self, id: &Identifier) {
        if self.ord == Ordering::Less || self.ord == Ordering::Equal {
            self.run_jmp(id);
        } else {
            self.inc_pointer();
        }
    }

    fn run_jl(&mut self, id: &Identifier) {
        if self.ord == Ordering::Less {
            self.run_jmp(id);
        } else {
            self.inc_pointer();
        }
    }

    fn run_call(&mut self, id: &Identifier) {
        self.call_stack.push(self.program_pointer);
        self.run_jmp(id);
    }

    fn run_ret(&mut self) {
        self.program_pointer = self.call_stack.pop().unwrap() + 1;
    }

    fn run_msg(&mut self, items: &[MsgItem]) {
        let mut s = String::new();
        for i in items {
            let string = match i {
                MsgItem::Str(s) => s.to_string(),
                MsgItem::Reg(key) => self.reg_value(&Reg {key: *key}).to_string()
            };
            s.push_str(&string);
        }

        self.output.push_str(&s);
        self.inc_pointer();
    }

    fn run_end(&mut self) {
        self.end = true;
    }

    fn check_labels(&mut self) {
        use Instruction::*;
        for (i, instr) in self.instructions.iter().enumerate() {
            if let Label(id) = instr {
                self.labels.insert(id.to_string(), i);
            }
        }
    }

    fn inc_pointer(&mut self) {
        self.program_pointer += 1;
    }

    fn reg_value(&self, r: &Reg) -> i32 {
        *self.registers.get(&r.key).unwrap()
    }

    fn modify_reg<F>(&mut self, r: &Reg, f: F)
    where
        F: Fn(i32) -> i32
    {
        let entry = self.registers.entry(r.key).or_insert(0);
        *entry = f(*entry);
    }

    fn modify_reg_by_eval<F>(&mut self, r: &Reg, e: &Eval, f: F)
    where
        F: Fn(i32, i32) -> i32
    {
        let eval = self.eval_value(e);
        let entry = self.registers.entry(r.key).or_insert(0);
        *entry = f(*entry, eval);
    }

    fn eval_value(&self, eval: &Eval) -> i32 {
        match eval {
            Eval::Reg(ch) => {
                *self.registers.get(ch).unwrap()
            },
            Eval::Num(n) => *n
        }
    }
}

pub mod parser {
    use super::*;
    use super::combinators::*;
    use super::parsers::*;

    pub fn parse(input: &str) -> Vec<Instruction> {
        let mut instructions = vec![];
        let mut input = input;

        while let Some((next_input, instruction)) = instruction().parse(input) {
            instructions.push(instruction);
            input = next_input;
        }

        instructions
    }

    fn instruction<'a>() -> impl Parser<'a, Instruction> {
        ignore_spaces(
            mov()
                .or(inc())
                .or(dec())
                .or(add())
                .or(sub())
                .or(mul())
                .or(div())
                .or(cmp())
                .or(jmp())
                .or(jne())
                .or(je())
                .or(jge())
                .or(jg())
                .or(jle())
                .or(jl())
                .or(call())
                .or(ret())
                .or(msg())
                .or(end())
                .or(comment())
                .or(label())
        )
    }

    fn mov<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;

        string("mov")
            .right(space1())
            .right(reg_and_eval())
            .map(|(reg, eval)| Mov(reg, eval))
    }

    fn inc<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;

        string("inc")
            .right(space1())
            .right(reg())
            .map(|r| Inc(r))
    }

    fn dec<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;

        string("dec")
            .right(space1())
            .right(reg())
            .map(|r| Dec(r))
    }

    fn add<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;

        string("add")
            .right(space1())
            .right(reg_and_eval())
            .map(|(reg, eval)| Add(reg, eval))
    }

    fn sub<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;

        string("sub")
            .right(space1())
            .right(reg_and_eval())
            .map(|(reg, eval)| Sub(reg, eval))
    }

    fn mul<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;

        string("mul")
            .right(space1())
            .right(reg_and_eval())
            .map(|(reg, eval)| Mul(reg, eval))
            .debug("mul")
    }

    fn div<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;

        string("div")
            .right(space1())
            .right(reg_and_eval())
            .map(|(reg, eval)| Div(reg, eval))
    }

    fn label<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;

        identifier()
            .left(char(':'))
            .map(|id| Label(id))
    }

    fn cmp<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;

        string("cmp")
            .right(space1())
            .right(eval())
            .left(arg_del())
            .pair(eval())
            .map(|(e1, e2)| Cmp(e1, e2))
    }

    fn jmp<'a>() -> impl Parser<'a, Instruction> {
        jump("jmp", Instruction::Jmp)
    }

    fn jne<'a>() -> impl Parser<'a, Instruction> {
        jump("jne", Instruction::Jne)
    }

    fn je<'a>() -> impl Parser<'a, Instruction> {
        jump("je", Instruction::Je)
    }

    fn jge<'a>() -> impl Parser<'a, Instruction> {
        jump("jge", Instruction::Jge)
    }

    fn jg<'a>() -> impl Parser<'a, Instruction> {
        jump("jg", Instruction::Jg)
    }

    fn jle<'a>() -> impl Parser<'a, Instruction> {
        jump("jle", Instruction::Jle)
    }

    fn jl<'a>() -> impl Parser<'a, Instruction> {
        jump("jl", Instruction::Jl)
    }

    fn call<'a>() -> impl Parser<'a, Instruction> {
        jump("call", Instruction::Call)
    }

    fn ret<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;
        string("ret")
            .map(|_| Ret)
    }

    fn msg<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;
        string("msg")
            .right(space1())
            .right(msg_item())
            .pair(
                zero_or_more(
                    arg_del().right(msg_item())
                )
            )
            .map(|(head, mut tail)| {
                tail.insert(0, head);
                Msg(tail)
            })
    }

    fn msg_item<'a>() -> impl Parser<'a, MsgItem> {
        let s = char('\'')
            .right(zero_or_more(any_char.pred(|ch| *ch != '\'')))
            .left(char('\''))
            .map(|chs| MsgItem::Str(chs.iter().collect()));

        let reg = any_char.map(MsgItem::Reg);

        s.or(reg)
    }

    fn end<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;

        string("end")
            .map(|_| End)
    }

    fn comment<'a>() -> impl Parser<'a, Instruction> {
        use Instruction::*;
        char(';')
            .pair(zero_or_more(any_char.pred(|ch| *ch != '\n')))
            .map(|_| Nop)
    }

    fn jump<'a, F>(cmd: &'a str, f: F)
        -> impl Parser<'a, Instruction>
    where
        F: Fn(Identifier) -> Instruction,
        F: 'a
    {
        string(cmd)
            .pair(space1())
            .right(identifier())
            .map(f)
    }

    fn reg_and_eval<'a>() -> impl Parser<'a, (Reg, Eval)> {
        reg()
            .left(arg_del())
            .pair(eval())
    }

    fn identifier<'a>() -> impl Parser<'a, Identifier> {
        one_or_more(
            any_char.pred(|ch| *ch != ':' && *ch != '\n' && *ch != ' ' && *ch != ';')
        ).map(|chs| chs.iter().collect())
    }

    fn arg_del<'a>() -> impl Parser<'a, ()> {
        ignore_spaces(char(','))
            .map(|_| ())
    }

    fn reg<'a>() -> impl Parser<'a, Reg> {
        any_char
            .map(|key| Reg {key})
    }

    fn eval<'a>() -> impl Parser<'a, Eval> {
        let reg = any_char.map(|ch| Eval::Reg(ch));
        let num = number.map(|num| Eval::Num(num));

        num.or(reg)
    }

}

pub mod parsers {
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
}

pub mod combinators {
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

}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn simple_test() {
        let simple_programs = &[
            "\n; My first program\nmov  a, 5\ninc  a\ncall function\nmsg  '(5+1)/2 = ', a    ; output message\nend\n\nfunction:\n    div  a, 2\n    ret\n",
            "\nmov   a, 5\nmov   b, a\nmov   c, a\ncall  proc_fact\ncall  print\nend\n\nproc_fact:\n    dec   b\n    mul   c, b\n    cmp   b, 1\n    jne   proc_fact\n    ret\n\nprint:\n    msg   a, '! = ', c ; output text\n    ret\n",
            "\nmov   a, 8            ; value\nmov   b, 0            ; next\nmov   c, 0            ; counter\nmov   d, 0            ; first\nmov   e, 1            ; second\ncall  proc_fib\ncall  print\nend\n\nproc_fib:\n    cmp   c, 2\n    jl    func_0\n    mov   b, d\n    add   b, e\n    mov   d, e\n    mov   e, b\n    inc   c\n    cmp   c, a\n    jle   proc_fib\n    ret\n\nfunc_0:\n    mov   b, c\n    inc   c\n    jmp   proc_fib\n\nprint:\n    msg   'Term ', a, ' of Fibonacci series is: ', b        ; output text\n    ret\n",
            "\nmov   a, 11           ; value1\nmov   b, 3            ; value2\ncall  mod_func\nmsg   'mod(', a, ', ', b, ') = ', d        ; output\nend\n\n; Mod function\nmod_func:\n    mov   c, a        ; temp1\n    div   c, b\n    mul   c, b\n    mov   d, a        ; temp2\n    sub   d, c\n    ret\n",
            "\nmov   a, 81         ; value1\nmov   b, 153        ; value2\ncall  init\ncall  proc_gcd\ncall  print\nend\n\nproc_gcd:\n    cmp   c, d\n    jne   loop\n    ret\n\nloop:\n    cmp   c, d\n    jg    a_bigger\n    jmp   b_bigger\n\na_bigger:\n    sub   c, d\n    jmp   proc_gcd\n\nb_bigger:\n    sub   d, c\n    jmp   proc_gcd\n\ninit:\n    cmp   a, 0\n    jl    a_abs\n    cmp   b, 0\n    jl    b_abs\n    mov   c, a            ; temp1\n    mov   d, b            ; temp2\n    ret\n\na_abs:\n    mul   a, -1\n    jmp   init\n\nb_abs:\n    mul   b, -1\n    jmp   init\n\nprint:\n    msg   'gcd(', a, ', ', b, ') = ', c\n    ret\n",
            "\ncall  func1\ncall  print\nend\n\nfunc1:\n    call  func2\n    ret\n\nfunc2:\n    ret\n\nprint:\n    msg 'This program should return null'\n",
            "\nmov   a, 2            ; value1\nmov   b, 10           ; value2\nmov   c, a            ; temp1\nmov   d, b            ; temp2\ncall  proc_func\ncall  print\nend\n\nproc_func:\n    cmp   d, 1\n    je    continue\n    mul   c, a\n    dec   d\n    call  proc_func\n\ncontinue:\n    ret\n\nprint:\n    msg a, '^', b, ' = ', c\n    ret\n"];

        let expected = &[
            Some(String::from("(5+1)/2 = 3")),
            Some(String::from("5! = 120")),
            Some(String::from("Term 8 of Fibonacci series is: 21")),
            Some(String::from("mod(11, 3) = 2")),
            Some(String::from("gcd(81, 153) = 9")),
            None,
            Some(String::from("2^10 = 1024"))];

        for (prg, exp) in simple_programs.iter().zip(expected) {
            let actual = AssemblerInterpreter::interpret(*prg);
            assert_eq!(actual, *exp);
        }
    }
}
