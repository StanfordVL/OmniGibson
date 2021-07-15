(define (problem storing_the_groceries_0)
    (:domain igibson)

    (:objects
     	cereal.n.03_1 cereal.n.03_2 - cereal.n.03
    	countertop.n.01_1 - countertop.n.01
    	lettuce.n.03_1 lettuce.n.03_2 - lettuce.n.03
    	broccoli.n.02_1 broccoli.n.02_2 - broccoli.n.02
    	raspberry.n.02_1 raspberry.n.02_2 - raspberry.n.02
    	pork.n.01_1 pork.n.01_2 - pork.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop cereal.n.03_1 countertop.n.01_1) 
        (ontop cereal.n.03_2 countertop.n.01_1) 
        (ontop lettuce.n.03_1 countertop.n.01_1) 
        (ontop lettuce.n.03_2 countertop.n.01_1) 
        (ontop broccoli.n.02_1 countertop.n.01_1) 
        (ontop broccoli.n.02_2 countertop.n.01_1) 
        (ontop raspberry.n.02_1 countertop.n.01_1) 
        (ontop raspberry.n.02_2 countertop.n.01_1) 
        (ontop pork.n.01_1 countertop.n.01_1) 
        (ontop pork.n.01_2 countertop.n.01_1) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (inside ?cereal.n.03_1 ?cabinet.n.01_1) 
            (inside ?cereal.n.03_2 ?cabinet.n.01_1) 
            (nextto ?cereal.n.03_1 ?cereal.n.03_2) 
            (inside ?lettuce.n.03_1 ?electric_refrigerator.n.01_1) 
            (inside ?lettuce.n.03_2 ?electric_refrigerator.n.01_1) 
            (nextto ?lettuce.n.03_1 ?lettuce.n.03_2) 
            (inside ?broccoli.n.02_1 ?electric_refrigerator.n.01_1) 
            (inside ?broccoli.n.02_2 ?electric_refrigerator.n.01_1) 
            (nextto ?broccoli.n.02_1 ?broccoli.n.02_2) 
            (inside ?raspberry.n.02_1 ?electric_refrigerator.n.01_1) 
            (inside ?raspberry.n.02_2 ?electric_refrigerator.n.01_1) 
            (nextto ?raspberry.n.02_1 ?raspberry.n.02_2) 
            (inside ?pork.n.01_1 ?electric_refrigerator.n.01_1) 
            (inside ?pork.n.01_2 ?electric_refrigerator.n.01_1) 
            (nextto ?pork.n.01_1 ?pork.n.01_2)
        )
    )
)