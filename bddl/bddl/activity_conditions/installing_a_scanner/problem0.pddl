(define (problem installing_a_scanner_0)
    (:domain igibson)

    (:objects
     	scanner.n.02_1 - scanner.n.02
    	table.n.02_1 - table.n.02
    	floor.n.01_1 - floor.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop scanner.n.02_1 table.n.02_1) 
        (not 
            (toggled_on scanner.n.02_1)
        ) 
        (inroom table.n.02_1 home_office) 
        (inroom floor.n.01_1 home_office) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (toggled_on ?scanner.n.02_1) 
            (under ?scanner.n.02_1 ?table.n.02_1)
        )
    )
)