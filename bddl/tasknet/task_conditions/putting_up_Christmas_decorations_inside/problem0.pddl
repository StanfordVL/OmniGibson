(define (problem putting_up_Christmas_decorations_inside_0)
    (:domain igibson)

    (:objects
     	christmas_tree.n.05_1 - christmas_tree.n.05
    	floor.n.01_1 - floor.n.01
    	wreath.n.01_1 wreath.n.01_2 - wreath.n.01
    	tinsel.n.02_1 tinsel.n.02_2 - tinsel.n.02
    	bow.n.08_1 - bow.n.08
    	window.n.01_1 window.n.01_2 - window.n.01
        agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor christmas_tree.n.05_1 floor.n.01_1) 
        (onfloor wreath.n.01_1 floor.n.01_1) 
        (onfloor wreath.n.01_2 floor.n.01_1) 
        (onfloor tinsel.n.02_1 floor.n.01_1) 
        (onfloor tinsel.n.02_2 floor.n.01_1) 
        (onfloor bow.n.08_1 floor.n.01_1) 
        (inroom floor.n.01_1 living_room) 
        (inroom window.n.01_1 living_room) 
        (inroom window.n.01_2 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (nextto ?christmas_tree.n.05_1 ?window.n.01_1) 
            (touching ?wreath.n.01_1 ?window.n.01_1) 
            (touching ?wreath.n.01_2 ?window.n.01_2) 
            (ontop ?tinsel.n.02_1 ?christmas_tree.n.05_1) 
            (ontop ?tinsel.n.02_2 ?christmas_tree.n.05_1) 
            (ontop ?bow.n.08_1 ?christmas_tree.n.05_1)
        )
    )
)