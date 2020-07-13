pub type Identifier = String;

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

