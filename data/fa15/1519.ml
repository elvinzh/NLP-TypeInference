
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr;;

let buildAverage (e1,e2) = Average (e1, e2);;

let buildCosine e = Cosine e;;

let buildSine e = Sine e;;

let buildThresh (a,b,a_less,b_less) = Thresh (a, b, a_less, b_less);;

let buildTimes (e1,e2) = Times (e1, e2);;

let buildX () = VarX;;

let buildY () = VarY;;

let rec build (rand,depth) =
  if depth > 0
  then
    match rand (1, 7) with
    | 1 -> buildX ()
    | 2 -> buildY ()
    | 3 -> buildSine (build (rand, (depth - 1)))
    | 4 -> buildCosine (build (rand, (depth - 1)))
    | 5 ->
        buildAverage
          ((build (rand, (depth - 1))), (build (rand, (depth - 1))))
    | 6 ->
        buildTimes ((build (rand, (depth - 1))), (build (rand, (depth - 1))))
    | 7 ->
        buildThresh
          ((build (rand, (depth - 1))), (build (rand, (depth - 1))),
            (build (rand, (depth - 1))));;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr;;

let buildAverage (e1,e2) = Average (e1, e2);;

let buildCosine e = Cosine e;;

let buildSine e = Sine e;;

let buildThresh (a,b,a_less,b_less) = Thresh (a, b, a_less, b_less);;

let buildTimes (e1,e2) = Times (e1, e2);;

let buildX () = VarX;;

let buildY () = VarY;;

let rec build (rand,depth) =
  if depth > 0
  then
    match rand (1, 5) with
    | 1 -> buildSine (build (rand, (depth - 1)))
    | 2 -> buildCosine (build (rand, (depth - 1)))
    | 3 ->
        buildAverage
          ((build (rand, (depth - 1))), (build (rand, (depth - 1))))
    | 4 ->
        buildTimes ((build (rand, (depth - 1))), (build (rand, (depth - 1))))
    | 5 ->
        buildThresh
          ((build (rand, (depth - 1))), (build (rand, (depth - 1))),
            (build (rand, (depth - 1))), (build (rand, (depth - 1))))
  else (match rand (1, 2) with | 1 -> buildX () | 2 -> buildY ());;

*)

(* changed spans
(26,2)-(41,40)
(28,4)-(41,40)
(28,19)-(28,20)
(29,11)-(29,17)
(29,11)-(29,20)
(29,18)-(29,20)
(30,11)-(30,17)
(30,11)-(30,20)
(30,18)-(30,20)
(40,10)-(41,40)
*)

(* type error slice
(17,3)-(17,69)
(17,17)-(17,67)
(21,3)-(21,22)
(21,11)-(21,20)
(21,16)-(21,20)
(26,2)-(41,40)
(28,4)-(41,40)
(29,11)-(29,17)
(29,11)-(29,20)
(39,8)-(39,19)
(39,8)-(41,40)
(40,10)-(41,40)
*)

(* all spans
(11,18)-(11,43)
(11,27)-(11,43)
(11,36)-(11,38)
(11,40)-(11,42)
(13,16)-(13,28)
(13,20)-(13,28)
(13,27)-(13,28)
(15,14)-(15,24)
(15,18)-(15,24)
(15,23)-(15,24)
(17,17)-(17,67)
(17,38)-(17,67)
(17,46)-(17,47)
(17,49)-(17,50)
(17,52)-(17,58)
(17,60)-(17,66)
(19,16)-(19,39)
(19,25)-(19,39)
(19,32)-(19,34)
(19,36)-(19,38)
(21,11)-(21,20)
(21,16)-(21,20)
(23,11)-(23,20)
(23,16)-(23,20)
(25,15)-(41,40)
(26,2)-(41,40)
(26,5)-(26,14)
(26,5)-(26,10)
(26,13)-(26,14)
(28,4)-(41,40)
(28,10)-(28,21)
(28,10)-(28,14)
(28,15)-(28,21)
(28,16)-(28,17)
(28,19)-(28,20)
(29,11)-(29,20)
(29,11)-(29,17)
(29,18)-(29,20)
(30,11)-(30,20)
(30,11)-(30,17)
(30,18)-(30,20)
(31,11)-(31,48)
(31,11)-(31,20)
(31,21)-(31,48)
(31,22)-(31,27)
(31,28)-(31,47)
(31,29)-(31,33)
(31,35)-(31,46)
(31,36)-(31,41)
(31,44)-(31,45)
(32,11)-(32,50)
(32,11)-(32,22)
(32,23)-(32,50)
(32,24)-(32,29)
(32,30)-(32,49)
(32,31)-(32,35)
(32,37)-(32,48)
(32,38)-(32,43)
(32,46)-(32,47)
(34,8)-(35,68)
(34,8)-(34,20)
(35,10)-(35,68)
(35,11)-(35,38)
(35,12)-(35,17)
(35,18)-(35,37)
(35,19)-(35,23)
(35,25)-(35,36)
(35,26)-(35,31)
(35,34)-(35,35)
(35,40)-(35,67)
(35,41)-(35,46)
(35,47)-(35,66)
(35,48)-(35,52)
(35,54)-(35,65)
(35,55)-(35,60)
(35,63)-(35,64)
(37,8)-(37,77)
(37,8)-(37,18)
(37,19)-(37,77)
(37,20)-(37,47)
(37,21)-(37,26)
(37,27)-(37,46)
(37,28)-(37,32)
(37,34)-(37,45)
(37,35)-(37,40)
(37,43)-(37,44)
(37,49)-(37,76)
(37,50)-(37,55)
(37,56)-(37,75)
(37,57)-(37,61)
(37,63)-(37,74)
(37,64)-(37,69)
(37,72)-(37,73)
(39,8)-(41,40)
(39,8)-(39,19)
(40,10)-(41,40)
(40,11)-(40,38)
(40,12)-(40,17)
(40,18)-(40,37)
(40,19)-(40,23)
(40,25)-(40,36)
(40,26)-(40,31)
(40,34)-(40,35)
(40,40)-(40,67)
(40,41)-(40,46)
(40,47)-(40,66)
(40,48)-(40,52)
(40,54)-(40,65)
(40,55)-(40,60)
(40,63)-(40,64)
(41,12)-(41,39)
(41,13)-(41,18)
(41,19)-(41,38)
(41,20)-(41,24)
(41,26)-(41,37)
(41,27)-(41,32)
(41,35)-(41,36)
*)