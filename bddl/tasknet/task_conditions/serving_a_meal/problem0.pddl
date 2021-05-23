(define (problem serving_a_meal_0)
    (:domain igibson)

    (:objects
     	chicken.n.01_1 chicken.n.01_2 - chicken.n.01
    	knife.n.01_1 knife.n.01_2 - knife.n.01
    	fork.n.01_1 fork.n.01_2 - fork.n.01
    	spoon.n.01_1 spoon.n.01_2 - spoon.n.01
    	plate.n.04_1 plate.n.04_2 - plate.n.04
    	soup.n.01_1 soup.n.01_2 - soup.n.01
    	table.n.02_1 - table.n.02
    	bread.n.01_1 bread.n.01_2 - bread.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	salad.n.01_1 salad.n.01_2 - salad.n.01
    	water.n.06_1 water.n.06_2 - water.n.06
    	cake.n.03_1 cake.n.03_2 - cake.n.03
    	stove.n.01_1 - stove.n.01
            cabinet.n.01_1 cabinet.n.01_2 - cabinet.n.01
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (cooked chicken.n.01_1) 
        (cooked chicken.n.01_2) 
        (inside knife.n.01_1 cabinet.n.01_1) 
        (inside knife.n.01_2 cabinet.n.01_1) 
        (inside fork.n.01_1 cabinet.n.01_1) 
        (inside fork.n.01_2 cabinet.n.01_1) 
        (inside spoon.n.01_1 cabinet.n.01_1) 
        (inside spoon.n.01_2 cabinet.n.01_1) 
        (inside plate.n.04_1 cabinet.n.01_2) 
        (inside plate.n.04_2 cabinet.n.01_1) 
        (ontop soup.n.01_1 table.n.02_1) 
        (ontop soup.n.01_2 table.n.02_1) 
        (inside bread.n.01_2 electric_refrigerator.n.01_1) 
        (inside bread.n.01_1 electric_refrigerator.n.01_1) 
        (inside salad.n.01_1 electric_refrigerator.n.01_1) 
        (inside salad.n.01_2 electric_refrigerator.n.01_1) 
        (inside chicken.n.01_1 electric_refrigerator.n.01_1) 
        (inside chicken.n.01_2 electric_refrigerator.n.01_1) 
        (inside water.n.06_1 electric_refrigerator.n.01_1) 
        (inside water.n.06_2 electric_refrigerator.n.01_1) 
        (inside cake.n.03_1 electric_refrigerator.n.01_1) 
        (inside cake.n.03_2 electric_refrigerator.n.01_1) 
        (inroom table.n.02_1 dining_room) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom stove.n.01_1 kitchen) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom cabinet.n.01_2 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (forall 
                (?plate.n.04 - plate.n.04) 
                (ontop ?plate.n.04 ?table.n.02_1)
            ) 
            (forall 
                (?knife.n.01 - knife.n.01) 
                (ontop ?knife.n.01 ?table.n.02_1)
            ) 
            (forall 
                (?fork.n.01 - fork.n.01) 
                (ontop ?fork.n.01 ?table.n.02_1)
            ) 
            (forpairs 
                (?spoon.n.01 - spoon.n.01) 
                (?soup.n.01 - soup.n.01) 
                (nextto ?spoon.n.01 ?soup.n.01)
            ) 
            (forall 
                (?water.n.06 - water.n.06) 
                (ontop ?water.n.06 ?table.n.02_1)
            ) 
            (forpairs 
                (?chicken.n.01 - chicken.n.01) 
                (?plate.n.04 - plate.n.04) 
                (ontop ?chicken.n.01 ?plate.n.04)
            ) 
            (forpairs 
                (?salad.n.01 - salad.n.01) 
                (?plate.n.04 - plate.n.04) 
                (nextto ?salad.n.01 ?plate.n.04)
            ) 
            (forpairs 
                (?bread.n.01 - bread.n.01) 
                (?plate.n.04 - plate.n.04) 
                (nextto ?bread.n.01 ?plate.n.04)
            ) 
            (forpairs 
                (?cake.n.03 - cake.n.03) 
                (?plate.n.04 - plate.n.04) 
                (nextto ?cake.n.03 ?plate.n.04)
            )
        )
    )
)