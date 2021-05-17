(define (problem sorting_groceries_0)
    (:domain igibson)

    (:objects
     	bread.n.01_1 - bread.n.01
    	countertop.n.01_1 - countertop.n.01
    	flour.n.01_1 - flour.n.01
    	floor.n.01_1 - floor.n.01
    	milk.n.01_1 - milk.n.01
    	chair.n.01_1 - chair.n.01
    	meat.n.01_1 - meat.n.01
    	table.n.02_1 - table.n.02
    	cheese.n.01_1 - cheese.n.01
    	yogurt.n.01_1 - yogurt.n.01
    	soup.n.01_1 - soup.n.01
    	carrot.n.03_1 carrot.n.03_2 carrot.n.03_3 - carrot.n.03
    	broccoli.n.02_1 - broccoli.n.02
    	apple.n.01_1 apple.n.01_2 - apple.n.01
    	orange.n.01_1 orange.n.01_2 orange.n.01_3 - orange.n.01
    	cabinet.n.01_1 - cabinet.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop bread.n.01_1 countertop.n.01_1) 
        (onfloor flour.n.01_1 floor.n.01_1) 
        (ontop milk.n.01_1 chair.n.01_1) 
        (ontop meat.n.01_1 table.n.02_1) 
        (ontop cheese.n.01_1 table.n.02_1) 
        (ontop yogurt.n.01_1 table.n.02_1) 
        (ontop soup.n.01_1 table.n.02_1) 
        (ontop carrot.n.03_1 countertop.n.01_1) 
        (ontop carrot.n.03_2 countertop.n.01_1) 
        (ontop carrot.n.03_3 countertop.n.01_1) 
        (ontop broccoli.n.02_1 countertop.n.01_1) 
        (ontop apple.n.01_1 countertop.n.01_1) 
        (ontop apple.n.01_2 countertop.n.01_1) 
        (ontop orange.n.01_1 table.n.02_1) 
        (ontop orange.n.01_2 table.n.02_1) 
        (ontop orange.n.01_3 table.n.02_1) 
        (inroom cabinet.n.01_1 kitchen) 
        (inroom countertop.n.01_1 kitchen) 
        (inroom table.n.02_1 kitchen) 
        (inroom chair.n.01_1 kitchen) 
        (inroom floor.n.01_1 kitchen) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (inside ?bread.n.01_1 ?cabinet.n.01_1) 
            (inside ?flour.n.01_1 ?cabinet.n.01_1) 
            (and 
                (inside ?milk.n.01_1 ?electric_refrigerator.n.01_1) 
                (inside ?meat.n.01_1 ?electric_refrigerator.n.01_1) 
                (inside ?cheese.n.01_1 ?electric_refrigerator.n.01_1) 
                (inside ?yogurt.n.01_1 ?electric_refrigerator.n.01_1) 
                (inside ?soup.n.01_1 ?electric_refrigerator.n.01_1)
            ) 
            (forall 
                (?carrot.n.03 - carrot.n.03) 
                (and 
                    (inside ?carrot.n.03 ?electric_refrigerator.n.01_1) 
                    (or 
                        (nextto ?carrot.n.03 ?carrot.n.03_1) 
                        (nextto ?carrot.n.03 ?carrot.n.03_2) 
                        (nextto ?carrot.n.03 ?carrot.n.03_3)
                    )
                )
            ) 
            (inside ?broccoli.n.02_1 ?electric_refrigerator.n.01_1) 
            (forall 
                (?apple.n.01 - apple.n.01) 
                (and 
                    (inside ?apple.n.01 ?electric_refrigerator.n.01_1) 
                    (or 
                        (nextto ?apple.n.01 ?apple.n.01_1) 
                        (nextto ?apple.n.01 ?apple.n.01_2)
                    )
                )
            ) 
            (forall 
                (?orange.n.01 - orange.n.01) 
                (and 
                    (ontop ?orange.n.01 ?table.n.02_1) 
                    (or 
                        (nextto ?orange.n.01 ?orange.n.01_1) 
                        (nextto ?orange.n.01 ?orange.n.01_2) 
                        (nextto ?orange.n.01 ?orange.n.01_3)
                    )
                )
            )
        )
    )
)