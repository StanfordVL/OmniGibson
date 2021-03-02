(define (problem cleaning_carpets_1)
    (:domain igibson)

    (:objects
     	vacuum1 - vacuum
    	carpet1 carpet2 - carpet
    	rag1 rag2 - rag
    	detergent1 detergent2 - detergent
    	soap1 soap2 - soap
    	floor1 floor2 - floor
    	shampoo1 shampoo2 - shampoo
    	bucket1 - bucket
    	mold1 mold2 mold3 mold4 mold5 mold6 - mold
    )
    
    (:init 
        (ontop vacuum1 carpet1) 
        (ontop rag1 carpet1) 
        (ontop rag2 carpet2) 
        (ontop detergent1 carpet1) 
        (ontop detergent2 carpet2) 
        (ontop soap1 carpet1) 
        (ontop soap2 carpet2) 
        (ontop carpet1 floor1) 
        (ontop shampoo1 carpet1) 
        (ontop shampoo2 carpet2) 
        (ontop carpet2 floor2) 
        (ontop bucket1 floor1) 
        (ontop mold1 carpet1) 
        (ontop mold2 carpet1) 
        (ontop mold3 carpet1) 
        (ontop mold4 carpet2) 
        (ontop mold5 carpet2) 
        (ontop mold6 carpet2) 
        (dusty carpet1) 
        (dusty carpet2) 
        (inroom carpet1 bedroom) 
        (inroom carpet2 livingroom) 
        (inroom floor1 bedroom) 
        (inroom floor2 livingroom)
    )
    
    (:goal 
        (and 
            (forall 
                (?mold - mold) 
                (inside ?mold ?bucket1)
            ) 
            (forall 
                (?shampoo - shampoo) 
                (inside ?shampoo ?bucket1)
            ) 
            (scrubbed ?carpet1) 
            (scrubbed ?carpet2)
        )
    )
)