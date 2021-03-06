
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine ex -> sin (pi *. (eval (ex, x, y)))
  | Cosine ex -> cos (pi *. (eval (ex, x, y)))
  | Average (ex1,ex2) -> ((eval (ex1, x, y)) +. (eval (ex2, x, y))) /. 2.
  | Times (ex1,ex2) -> (eval (ex1, x, y)) * (eval (ex2, x, y))
  | Thresh (ex1,ex2,ex3,ex4) ->
      if (eval (ex1, x, y)) < (eval (ex2, x, y))
      then eval (ex3, x, y)
      else eval (ex4, x, y);;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine ex -> sin (pi *. (eval (ex, x, y)))
  | Cosine ex -> cos (pi *. (eval (ex, x, y)))
  | Average (ex1,ex2) -> ((eval (ex1, x, y)) +. (eval (ex2, x, y))) /. 2.
  | Times (ex1,ex2) -> (eval (ex1, x, y)) *. (eval (ex2, x, y))
  | Thresh (ex1,ex2,ex3,ex4) ->
      if (eval (ex1, x, y)) < (eval (ex2, x, y))
      then eval (ex3, x, y)
      else eval (ex4, x, y);;

*)

(* changed spans
(20,23)-(20,62)
*)

(* type error slice
(17,19)-(17,44)
(17,26)-(17,43)
(17,27)-(17,31)
(20,23)-(20,41)
(20,23)-(20,62)
(20,24)-(20,28)
(20,44)-(20,62)
(20,45)-(20,49)
*)

(* all spans
(11,9)-(11,26)
(11,9)-(11,12)
(11,16)-(11,26)
(11,17)-(11,21)
(11,22)-(11,25)
(13,14)-(24,27)
(14,2)-(24,27)
(14,8)-(14,9)
(15,13)-(15,14)
(16,13)-(16,14)
(17,15)-(17,44)
(17,15)-(17,18)
(17,19)-(17,44)
(17,20)-(17,22)
(17,26)-(17,43)
(17,27)-(17,31)
(17,32)-(17,42)
(17,33)-(17,35)
(17,37)-(17,38)
(17,40)-(17,41)
(18,17)-(18,46)
(18,17)-(18,20)
(18,21)-(18,46)
(18,22)-(18,24)
(18,28)-(18,45)
(18,29)-(18,33)
(18,34)-(18,44)
(18,35)-(18,37)
(18,39)-(18,40)
(18,42)-(18,43)
(19,25)-(19,73)
(19,25)-(19,67)
(19,26)-(19,44)
(19,27)-(19,31)
(19,32)-(19,43)
(19,33)-(19,36)
(19,38)-(19,39)
(19,41)-(19,42)
(19,48)-(19,66)
(19,49)-(19,53)
(19,54)-(19,65)
(19,55)-(19,58)
(19,60)-(19,61)
(19,63)-(19,64)
(19,71)-(19,73)
(20,23)-(20,62)
(20,23)-(20,41)
(20,24)-(20,28)
(20,29)-(20,40)
(20,30)-(20,33)
(20,35)-(20,36)
(20,38)-(20,39)
(20,44)-(20,62)
(20,45)-(20,49)
(20,50)-(20,61)
(20,51)-(20,54)
(20,56)-(20,57)
(20,59)-(20,60)
(22,6)-(24,27)
(22,9)-(22,48)
(22,9)-(22,27)
(22,10)-(22,14)
(22,15)-(22,26)
(22,16)-(22,19)
(22,21)-(22,22)
(22,24)-(22,25)
(22,30)-(22,48)
(22,31)-(22,35)
(22,36)-(22,47)
(22,37)-(22,40)
(22,42)-(22,43)
(22,45)-(22,46)
(23,11)-(23,27)
(23,11)-(23,15)
(23,16)-(23,27)
(23,17)-(23,20)
(23,22)-(23,23)
(23,25)-(23,26)
(24,11)-(24,27)
(24,11)-(24,15)
(24,16)-(24,27)
(24,17)-(24,20)
(24,22)-(24,23)
(24,25)-(24,26)
*)
