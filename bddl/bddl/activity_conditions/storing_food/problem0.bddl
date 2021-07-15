(define (problem storing_food_0)
    (:domain igibson)

    (:objects
     	oatmeal.n.01_1 oatmeal.n.01_2 - oatmeal.n.01
    	countertop.n.01_1 - countertop.n.01
    	chip.n.04_1 chip.n.04_2 - chip.n.04
    	vegetable_oil.n.01_1 vegetable_oil.n.01_2 - vegetable_oil.n.01
    	sugar.n.01_1 sugar.n.01_2 - sugar.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop oatmeal.n.01_1 countertop.n.01_1) 
        (ontop oatmeal.n.01_2 countertop.n.01_1) 
        (ontop chip.n.04_1 countertop.n.01_1) 
        (ontop chip.n.04_2 countertop.n.01_1) 
        (ontop vegetable_oil.n.01_1 countertop.n.01_1) 
        (ontop vegetable_oil.n.01_2 countertop.n.01_1) 
        (ontop sugar.n.01_1 countertop.n.01_1) 
        (ontop sugar.n.01_2 countertop.n.01_1) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (inside ?oatmeal.n.01_1 ?cabinet.n.01_1) 
            (inside ?oatmeal.n.01_2 ?cabinet.n.01_1) 
            (inside ?chip.n.04_1 ?cabinet.n.01_1) 
            (inside ?chip.n.04_2 ?cabinet.n.01_1) 
            (inside ?vegetable_oil.n.01_1 ?cabinet.n.01_1) 
            (inside ?vegetable_oil.n.01_2 ?cabinet.n.01_1) 
            (inside ?sugar.n.01_1 ?cabinet.n.01_1) 
            (inside ?sugar.n.01_2 ?cabinet.n.01_1)
        )
    )
)