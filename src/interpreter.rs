use std::collections::HashMap;
use std::cmp::Ordering;
use super::structs::*;

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

