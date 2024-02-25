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

    def getQuestion(self) -> list:
        return self.rules['question']

    def addRule(self, ruleName : str, patterns : list, responses : list, update : bool) -> None:
        new_rule = {
            "ruleName": ruleName,
            "patterns": patterns,
            "responses": responses
        }
        if update:
            self.rules['rules'].append(new_rule)

        try:
            with open(self.filename, 'r+') as file:
                file_data = json.load(file)
                file_data['rules'].append(new_rule)
                file.seek(0)
                json.dump(file_data, file, indent=4)
                print(f'Rule "{ruleName}" added successfully.')
        except Exception as e:
            print(f'Error while Writing to {self.filename}: {e}')
