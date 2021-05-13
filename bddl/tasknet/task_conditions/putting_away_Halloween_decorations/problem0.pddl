(define (problem putting_away_Halloween_decorations_0)
    (:domain igibson)

    (:objects
     	pumpkin.n.02_1 - pumpkin.n.02
    	floor.n.01_1 - floor.n.01
    	candle.n.01_1 candle.n.01_2 candle.n.01_3 candle.n.01_4 candle.n.01_5 - candle.n.01
    	caldron.n.01_1 - caldron.n.01
    	sheet.n.03_1 - sheet.n.03
    	sofa.n.01_1 - sofa.n.01
    	box.n.01_1 - box.n.01
    	window.n.01_1 - window.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (onfloor pumpkin.n.02_1 floor.n.01_1) 
        (onfloor candle.n.01_1 floor.n.01_1) 
        (onfloor candle.n.01_1 floor.n.01_1) 
        (onfloor candle.n.01_2 floor.n.01_1) 
        (onfloor candle.n.01_3 floor.n.01_1) 
        (onfloor candle.n.01_4 floor.n.01_1) 
        (onfloor candle.n.01_5 floor.n.01_1) 
        (onfloor caldron.n.01_1 floor.n.01_1) 
        (ontop sheet.n.03_1 sofa.n.01_1) 
        (onfloor box.n.01_1 floor.n.01_1) 
        (inroom window.n.01_1 living_room) 
        (inroom floor.n.01_1 living_room) 
        (inroom sofa.n.01_1 living_room) 
        (onfloor agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (ontop ?pumpkin.n.02_1 ?box.n.01_1) 
            (inside ?candle.n.01_1 ?box.n.01_1) 
            (inside ?candle.n.01_1 ?box.n.01_1) 
            (inside ?candle.n.01_2 ?box.n.01_1) 
            (inside ?candle.n.01_3 ?box.n.01_1) 
            (inside ?candle.n.01_4 ?box.n.01_1) 
            (inside ?candle.n.01_5 ?box.n.01_1) 
            (inside ?sheet.n.03_1 ?box.n.01_1) 
            (onfloor ?box.n.01_1 ?floor.n.01_1)
        )
    )
)