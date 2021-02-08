(define (problem cleaning_shoes_1)
    (:domain igibson)

    (:objects
     	faucet1 - faucet
    	sink1 - sink
    	water1 - water
    	bucket1 - bucket
    	floor1 - floor
    	drain1 - drain
    	cleansing_agent1 - cleansing_agent
    	rack1 - rack
    	brush1 - brush
    	counter1 - counter
    	shoelace1 shoelace2 shoelace3 shoelace4 - shoelace
    	boot1 boot2 - boot
    	gym_shoe1 gym_shoe2 - gym_shoe
    )
    
    (:init 
        (and 
            (nextto faucet1 sink1) 
            (and 
                (inside water1 faucet1) 
                (inside water1 bucket1) 
                (under floor1 water1)
            ) 
            (nextto sink1 water1) 
            (under drain1 sink1)
        ) 
        (and 
            (inside cleansing_agent1 bucket1) 
            (ontop bucket1 floor1) 
            (under floor1 cleansing_agent1) 
            (ontop rack1 floor1) 
            (ontop brush1 counter1)
        ) 
        (and 
            (and 
                (inside shoelace1 boot1) 
                (not 
                    (scrubbed shoelace1)
                ) 
                (ontop boot1 floor1) 
                (under floor1 shoelace1) 
                (dusty boot1) 
                (not 
                    (scrubbed boot1)
                )
            ) 
            (and 
                (inside shoelace2 boot2) 
                (not 
                    (scrubbed shoelace2)
                ) 
                (ontop boot2 floor1) 
                (under floor1 shoelace2) 
                (dusty boot2) 
                (not 
                    (scrubbed boot2)
                )
            ) 
            (and 
                (inside shoelace3 gym_shoe1) 
                (not 
                    (scrubbed shoelace3)
                ) 
                (ontop gym_shoe1 floor1) 
                (under floor1 shoelace3) 
                (dusty gym_shoe1) 
                (not 
                    (scrubbed gym_shoe1)
                )
            ) 
            (and 
                (inside shoelace4 gym_shoe2) 
                (not 
                    (scrubbed shoelace4)
                ) 
                (ontop gym_shoe2 floor1) 
                (under floor1 shoelace4) 
                (dusty gym_shoe2) 
                (not 
                    (scrubbed gym_shoe2)
                )
            )
        ) 
        (inroom floor1 bathroom) 
        (inroom sink1 bathroom) 
        (inroom counter1 bathroom)
    )
    
    (:goal 
        (and 
            (and 
                (nextto ?brush1 ?bucket1) 
                (ontop ?rack1 ?sink1) 
                (inside ?cleansing_agent1 ?drain1) 
                (inside ?water1 ?drain1) 
                (ontop ?bucket1 ?floor1)
            ) 
            (forall 
                (?shoelace - shoelace) 
                (and 
                    (ontop ?shoelace ?rack1) 
                    (scrubbed ?shoelace) 
                    (soaked ?shoelace)
                )
            ) 
            (forall 
                (?boot - boot) 
                (and 
                    (ontop ?boot ?rack1) 
                    (scrubbed ?boot) 
                    (not 
                        (dusty ?boot)
                    )
                )
            ) 
            (forall 
                (?gym_shoe - gym_shoe) 
                (and 
                    (ontop ?gym_shoe ?rack1) 
                    (scrubbed ?gym_shoe) 
                    (not 
                        (dusty ?gym_shoe)
                    )
                )
            )
        )
    )
)