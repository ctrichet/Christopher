from RulesList import RulesList

if __name__ == '__main__' :
    rules = RulesList('rules.json')
    rules.readRules()
    for rule in rules.getRules() :
        print(rule)
