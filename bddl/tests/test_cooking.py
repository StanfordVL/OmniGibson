import sys
from tasknet.parsing import parse_domain, parse_problem
from tasknet.condition_evaluation import compile_state, evaluate_state
import pprint

#################### TEST STUFF ####################

class Task(object):
    def __init__(self, obj_list):
        self.objects = obj_list

    def cooked(self, objA):
        return objA.iscooked


class Chicken(object):
    def __init__(self, obj_id, iscooked):
        self.category = 'chicken'
        self.iscooked = iscooked
        self.obj_id = obj_id


class Apple(object):
    def __init__(self, obj_id, iscooked):
        self.category = 'apple'
        self.iscooked = iscooked
        self.obj_id = obj_id

def test_chicken_cooking():
    atus_activity = 'checking_test'
    task_instance = 0
    domain_name, requirements, types, actions, predicates = parse_domain(
        atus_activity, task_instance)
    problem_name, objects, initial_state, goal_state = parse_problem(
        atus_activity, task_instance, domain_name)

    test_objects = [Chicken(1, False),
                    Chicken(2, False),
                    Chicken(3, False),
                    Chicken(4, False),
                    Apple(1, False),
                    Apple(2, False),
                    Apple(3, False)]

    test_task = Task(test_objects)

    scope_labels = ['chicken1', 'chicken2', 'chicken3',
                    'chicken4', 'apple1', 'apple2', 'apple3']
    test_scope = {label: obj for label, obj in zip(scope_labels, test_objects)}

    print('\n\nCompile conditions')
    compiled_state = compile_state(goal_state, test_task, scope=test_scope, object_map={'chicken': ['chicken1', 'chicken2', 'chicken3', 'chicken4'], 'apple': ['apple1', 'apple2', 'apple3']})

    print('Evaluate without action')
    success, results = evaluate_state(compiled_state)
    print('SUCCESS:', success)
    print('Satisfied conditions:', results['satisfied'])
    print('Unsatisfied conditions:', results['unsatisfied'])

    print('\n\nCook chicken1')
    test_scope['chicken1'].iscooked = True
    success, results = evaluate_state(compiled_state)
    print('SUCCESS:', success)
    print('Satisfied conditions:', results['satisfied'])
    print('Unsatisfied conditions:', results['unsatisfied'])

    print('\n\nCook chicken2-4')
    test_scope['chicken2'].iscooked = True
    test_scope['chicken3'].iscooked = True
    test_scope['chicken4'].iscooked = True
    success, results = evaluate_state(compiled_state)
    print('SUCCESS:', success)
    print('Satisfied conditions:', results['satisfied'])
    print('Unsatisfied conditions:', results['unsatisfied'])

    print('\n\nCook apple1')
    test_scope['apple1'].iscooked = True
    success, results = evaluate_state(compiled_state)
    print('SUCCESS:', success)
    print('Satisfied conditions:', results['satisfied'])
    print('Unsatisfied conditions:', results['unsatisfied'])

    print('\n\nCook apple2')
    test_scope['apple2'].iscooked = True
    success, results = evaluate_state(compiled_state)
    print('SUCCESS:', success)
    print('Satisfied conditions:', results['satisfied'])
    print('Unsatisfied conditions:', results['unsatisfied'])

    print("test")
    assert results['satisfied'] == [0, 2, 3, 4, 5], "Error, predicate evaluation returned wrong truth values"
    # print("Successful evaluation")

# def _test():
#     parsed_condition = ["and",
#                         ["forall",
#                          ["?chick", "-", "chicken"],
#                          ["cooked", "?ch"]
#                          ],
#                         ["or",
#                          ["exists",
#                           ["?ap", "-", "apple"],
#                           ["not",
#                            ["cooked", "?ap"]
#                            ]
#                           ],
#                          ["forall",
#                           ["?ap", "-", "apple"],
#                           ["cooked", "?ap"]
#                           ]
#                          ],
#                         ],
#     # ["imply",
#     #     ["cooked", "?ap"],
#     #     ["cooked", "?chick"]
#     # ]
#     # ]

#     parsed_condition2 = ["forall",
#                          ["?chick", "-", "chicken"],
#                          ["cooked", "?chick"]
#                          ]
#     # ]

#     parsed_condition3 = ["forall",
#                          ["?chick", "-", "chicken"],
#                          ["not", ["cooked", "?chick"]]
#                          ]
#     # ]

#     parsed_condition4 = ["exists",
#                          ["?chick", "-", "chicken"],
#                          ["cooked", "?chick"]
#                          ]
#     # ]

#     parsed_condition5 = ["exists",
#                          ["?chick", "-", "chicken"],
#                          ["not", ["cooked", "?chick"]]
#                          ]
#     # ]

#     parsed_condition6 = ["and",
#                          ["cooked", "?chick"],
#                          ["cooked", "?"]
#                          ]

#     # ]
#     parsed_condition7 = ["imply",
#                          ["not", ["cooked", "?ap"]],
#                          ["cooked", "?chick"]
#                          ]

#     # obj_list = [TestChicken(), TestApple(), TestChicken(), TestChicken()]
#     obj_list = [TestChickenCooked(), TestAppleUncooked(),
#                 TestChickenUncooked(), TestChickenCooked()]
#     task = Task(obj_list)

#     parsed_conditions = [
#         parsed_condition2,
#         parsed_condition3,
#         parsed_condition4,
#         parsed_condition5,
#         parsed_condition7
#     ]

#     for i, parsed_condition in enumerate(parsed_conditions):
#         print('CONDITION', i)
#         cond = HEAD({}, task, parsed_condition)
#         print('\nResolving...')
#         print('Result:', cond.evaluate())
#         print('\n')
