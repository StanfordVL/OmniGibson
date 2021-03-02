(define (problem cleaning_oven_0)
    (:domain igibson)

    (:objects
     	burner1 burner2 burner3 burner4 - burner
    	oven1 - oven
    	pan1 - pan
    	pot1 - pot
    	vegetable_oil1 - vegetable_oil
    	crumb1 crumb2 crumb3 crumb4 - crumb
    	tray1 - tray
    	sink1 - sink
    	scrub_brush1 - scrub_brush
    	faucet1 - faucet
    	water1 - water
    	rag1 - rag
    	soap1 - soap
    	bottle1 - bottle
    	garbage1 - garbage
    )
    
    (:init 
        (and 
            (and 
                (ontop burner1 oven1) 
                (dusty burner1) 
                (ontop pan1 burner1) 
                (under oven1 pan1)
            ) 
            (and 
                (ontop burner2 oven1) 
                (dusty burner2) 
                (ontop pot1 burner2) 
                (under oven1 pot1)
            ) 
            (and 
                (ontop burner3 oven1) 
                (dusty burner3) 
                (ontop vegetable_oil1 burner3) 
                (under oven1 vegetable_oil1)
            ) 
            (and 
                (ontop burner4 oven1) 
                (dusty burner4)
            ) 
            (and 
                (inside crumb1 oven1) 
                (burnt crumb1) 
                (inside crumb2 oven1) 
                (burnt crumb2) 
                (inside crumb3 oven1) 
                (burnt crumb3) 
                (inside crumb4 oven1) 
                (burnt crumb4)
            ) 
            (inside tray1 oven1) 
            (dusty oven1) 
            (not 
                (scrubbed oven1)
            )
        ) 
        (nextto sink1 oven1) 
        (and 
            (nextto scrub_brush1 sink1) 
            (and 
                (nextto faucet1 sink1) 
                (inside water1 faucet1) 
                (nextto sink1 water1)
            ) 
            (and 
                (nextto rag1 sink1) 
                (not 
                    (soaked rag1)
                )
            ) 
            (and 
                (inside soap1 bottle1) 
                (nextto bottle1 sink1) 
                (nextto sink1 soap1)
            ) 
            (nextto garbage1 sink1)
        ) 
        (inroom oven1 kitchen) 
        (inroom sink1 kitchen)
    )
    
    (:goal 
        (and 
            (and 
                (forall 
                    (?crumb - crumb) 
                    (inside ?crumb ?garbage1)
                ) 
                (forall 
                    (?burner - burner) 
                    (not 
                        (dusty ?burner)
                    )
                ) 
                (scrubbed ?oven1) 
                (not 
                    (dusty ?oven1)
                ) 
                (not 
                    (open ?oven1)
                )
            ) 
            (and 
                (and 
                    (inside ?pan1 ?sink1) 
                    (inside ?pot1 ?sink1) 
                    (inside ?tray1 ?sink1)
                ) 
                (and 
                    (inside ?water1 ?rag1) 
                    (soaked ?rag1)
                ) 
                (and 
                    (inside ?soap1 ?scrub_brush1) 
                    (soaked ?scrub_brush1)
                )
            )
        )
    )
)