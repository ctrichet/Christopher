from dataclasses import dataclass, field
from typing import List
from Rule import Rule
import json

@dataclass
class RulesList:
    filename    : str
    rules       : List[Rule] = field(default_factory=list)

    def readRules(self) -> None:
        try:
            with open(self.filename) as data:
                self.rules = json.load(data)
        except Exception as e:
            print(f'Error while Reading {self.filename} : {e}')

    def getRule(self) -> Rule:
        for rule in self.rules['rules']:
            yield Rule(rule['ruleName'], rule['patterns'], rule['responses'])

    def getUnknown(self) -> list:
        return self.rules['unknown']