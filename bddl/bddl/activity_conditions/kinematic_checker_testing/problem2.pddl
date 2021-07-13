(define (problem packlunch2)
    (:domain packlunch)
    (:objects 
        chips1 chips2 chips3 - chips
        fruit1 fruit2 fruit3 - fruit 
        container1 container2 container3 - container
        shelf1 - shelf 
        counter1 - counter
    )
    (:init 
        (inside chips1 shelf1)
        (inside chips2 shelf1)
        (inside chips3 shelf1)
        (inside fruit1 shelf1)
        (inside fruit2 shelf1)
        (inside fruit3 shelf1)
        (or (nextTo container1 container2)
            (nextTo container2 container3))
    )
    (:goal 
        (and (forall (?ch - chip) (exists (?con - container) (not (inside ?ch ?con))))
             (forall (?fr - fruit) (exists (?con - container) (inside ?fr ?con)))
             (forall (?con1 - container) (exists (?con2 - container) (nextTo ?con1 ?con2)))
             (not (inside chips1 shelf1))
        )
    )
)




; 3 fruit exist
; 3 chips exist
; 3 container exist
; 1 shelf exists 
; 1 counter exists 
; if chips then inside shelf 
; if fruit then inside shelf 
; if container then next to another container 

;    (:goal (imply (container ?containerX) (and (nextTo ?containerX ?containerY) (container ?containerY)))
;           (imply (chips ?chipsX) (inside ?chipsX ?containerX))
;           (imply (fruit ?fruitX) (inside ?fruitX ?containerX))
;           (imply (soda ?sodaX) (inside ?sodaX ?containerX)))
;    )