from dataclasses import dataclass, field
from typing import List

@dataclass
class Rule:
    ruleName    : str
    patterns    : List[str] = field(default_factory=list)
    responses   : List[str] = field(default_factory=list)

    def getRuleName(self) -> str:
        return self.ruleName

    def getPatterns(self) -> list:
        return self.patterns

    def getResponses(self) -> list:
        return self.responses