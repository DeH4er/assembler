use crate::structs::*;
use crate::combinators::*;

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

