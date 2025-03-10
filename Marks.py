from enum import Enum

class Mark(Enum):
    TAIL = 3   # "-"
    ARROW = 2  # ">"
    CIRCLE = 1 # "o"


class Edge:
    def __init__(self, start, end, lmark: Mark, rmark: Mark):
        self.start = start
        self.end = end
        self.lmark = lmark  # 这里的 mark 可能是 Mark.TAIL, Mark.ARROW, Mark.CIRCLE
        self.rmark = rmark
    def __repr__(self):
        rmark_symbol = {Mark.TAIL: "-", Mark.ARROW: ">", Mark.CIRCLE: "o"}
        lmark_symbol = {Mark.TAIL: "-", Mark.ARROW: "<", Mark.CIRCLE: "o"}
        return f"{self.start}{lmark_symbol[self.lmark]}-{rmark_symbol[self.rmark]} {self.end}"

if __name__ == "__main__":
    edge1 = Edge("A", "B", Mark.ARROW,Mark.TAIL)
    edge2 = Edge("B", "C", Mark.CIRCLE,Mark.CIRCLE)

    print(edge1)  # 输出: A > B
    print(edge2)  # 输出: B o C


