
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Divide of expr* expr
  | MultDiv of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine e1 -> sin (pi *. (eval (e1, x, y)))
  | Cosine e1 -> cos (pi *. (eval (e1, x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (e1,e2,e3,e4) when (eval (e1, x, y)) < (eval (e2, x, y)) ->
      eval (e3, x, y)
  | Thresh (e1,e2,e3,e4) -> eval (e4, x, y)
  | Divide (e1,e2) -> (eval (e1, x, y)) / (eval (e2, x, y))
  | MultDiv (e1,e2,e3) ->
      ((eval (e1, x, y)) * (eval (e2, x, y))) / (eval (e3, x, y));;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Divide of expr* expr
  | MultDiv of expr* expr* expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine e1 -> sin (pi *. (eval (e1, x, y)))
  | Cosine e1 -> cos (pi *. (eval (e1, x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (e1,e2,e3,e4) when (eval (e1, x, y)) < (eval (e2, x, y)) ->
      eval (e3, x, y)
  | Thresh (e1,e2,e3,e4) -> eval (e4, x, y)
  | Divide (e1,e2) -> (eval (e1, x, y)) /. (eval (e2, x, y))
  | MultDiv (e1,e2,e3) ->
      ((eval (e1, x, y)) *. (eval (e2, x, y))) /. (eval (e3, x, y));;

*)

(* changed spans
(26,22)-(26,59)
(26,23)-(26,27)
(28,6)-(28,45)
(28,6)-(28,65)
*)

(* type error slice
(19,19)-(19,44)
(19,26)-(19,43)
(19,27)-(19,31)
(26,22)-(26,39)
(26,22)-(26,59)
(26,23)-(26,27)
(26,42)-(26,59)
(26,43)-(26,47)
(28,6)-(28,45)
(28,6)-(28,65)
(28,7)-(28,24)
(28,8)-(28,12)
(28,27)-(28,44)
(28,28)-(28,32)
(28,48)-(28,65)
(28,49)-(28,53)
*)

(* all spans
(13,9)-(13,26)
(13,9)-(13,12)
(13,16)-(13,26)
(13,17)-(13,21)
(13,22)-(13,25)
(15,14)-(28,65)
(16,2)-(28,65)
(16,8)-(16,9)
(23,30)-(23,67)
(23,30)-(23,47)
(23,31)-(23,35)
(23,36)-(23,46)
(23,37)-(23,39)
(23,41)-(23,42)
(23,44)-(23,45)
(23,50)-(23,67)
(23,51)-(23,55)
(23,56)-(23,66)
(23,57)-(23,59)
(23,61)-(23,62)
(23,64)-(23,65)
(17,13)-(17,14)
(18,13)-(18,14)
(19,15)-(19,44)
(19,15)-(19,18)
(19,19)-(19,44)
(19,20)-(19,22)
(19,26)-(19,43)
(19,27)-(19,31)
(19,32)-(19,42)
(19,33)-(19,35)
(19,37)-(19,38)
(19,40)-(19,41)
(20,17)-(20,46)
(20,17)-(20,20)
(20,21)-(20,46)
(20,22)-(20,24)
(20,28)-(20,45)
(20,29)-(20,33)
(20,34)-(20,44)
(20,35)-(20,37)
(20,39)-(20,40)
(20,42)-(20,43)
(21,23)-(21,70)
(21,23)-(21,63)
(21,24)-(21,41)
(21,25)-(21,29)
(21,30)-(21,40)
(21,31)-(21,33)
(21,35)-(21,36)
(21,38)-(21,39)
(21,45)-(21,62)
(21,46)-(21,50)
(21,51)-(21,61)
(21,52)-(21,54)
(21,56)-(21,57)
(21,59)-(21,60)
(21,67)-(21,70)
(22,21)-(22,59)
(22,21)-(22,38)
(22,22)-(22,26)
(22,27)-(22,37)
(22,28)-(22,30)
(22,32)-(22,33)
(22,35)-(22,36)
(22,42)-(22,59)
(22,43)-(22,47)
(22,48)-(22,58)
(22,49)-(22,51)
(22,53)-(22,54)
(22,56)-(22,57)
(24,6)-(24,21)
(24,6)-(24,10)
(24,11)-(24,21)
(24,12)-(24,14)
(24,16)-(24,17)
(24,19)-(24,20)
(25,28)-(25,43)
(25,28)-(25,32)
(25,33)-(25,43)
(25,34)-(25,36)
(25,38)-(25,39)
(25,41)-(25,42)
(26,22)-(26,59)
(26,22)-(26,39)
(26,23)-(26,27)
(26,28)-(26,38)
(26,29)-(26,31)
(26,33)-(26,34)
(26,36)-(26,37)
(26,42)-(26,59)
(26,43)-(26,47)
(26,48)-(26,58)
(26,49)-(26,51)
(26,53)-(26,54)
(26,56)-(26,57)
(28,6)-(28,65)
(28,6)-(28,45)
(28,7)-(28,24)
(28,8)-(28,12)
(28,13)-(28,23)
(28,14)-(28,16)
(28,18)-(28,19)
(28,21)-(28,22)
(28,27)-(28,44)
(28,28)-(28,32)
(28,33)-(28,43)
(28,34)-(28,36)
(28,38)-(28,39)
(28,41)-(28,42)
(28,48)-(28,65)
(28,49)-(28,53)
(28,54)-(28,64)
(28,55)-(28,57)
(28,59)-(28,60)
(28,62)-(28,63)
*)