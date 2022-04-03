
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
  | VarX  -> 1.0 *. x
  | VarY  -> 1.0 *. y
  | Sine e' -> sin (pi *. (eval (e', x, y)))
  | Cosine e' -> cos (pi *. (eval (e', x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) + (eval (e2, x, y))) / 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (a,b,c,d) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (c, x, y)
      else eval (d, x, y);;


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
  | VarX  -> 1.0 *. x
  | VarY  -> 1.0 *. y
  | Sine e' -> sin (pi *. (eval (e', x, y)))
  | Cosine e' -> cos (pi *. (eval (e', x, y)))
  | Average (e1,e2) -> ((eval (e1, x, y)) +. (eval (e2, x, y))) /. 2.0
  | Times (e1,e2) -> (eval (e1, x, y)) *. (eval (e2, x, y))
  | Thresh (a,b,c,d) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (c, x, y)
      else eval (d, x, y);;

*)

(* changed spans
(19,23)-(19,62)
(19,23)-(19,68)
(19,24)-(19,41)
*)

(* type error slice
(14,2)-(24,25)
(15,13)-(15,21)
(17,19)-(17,44)
(17,26)-(17,43)
(17,27)-(17,31)
(19,23)-(19,62)
(19,23)-(19,68)
(19,24)-(19,41)
(19,25)-(19,29)
(19,44)-(19,61)
(19,45)-(19,49)
(19,65)-(19,68)
*)

(* all spans
(11,9)-(11,26)
(11,9)-(11,12)
(11,16)-(11,26)
(11,17)-(11,21)
(11,22)-(11,25)
(13,14)-(24,25)
(14,2)-(24,25)
(14,8)-(14,9)
(15,13)-(15,21)
(15,13)-(15,16)
(15,20)-(15,21)
(16,13)-(16,21)
(16,13)-(16,16)
(16,20)-(16,21)
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
(19,23)-(19,68)
(19,23)-(19,62)
(19,24)-(19,41)
(19,25)-(19,29)
(19,30)-(19,40)
(19,31)-(19,33)
(19,35)-(19,36)
(19,38)-(19,39)
(19,44)-(19,61)
(19,45)-(19,49)
(19,50)-(19,60)
(19,51)-(19,53)
(19,55)-(19,56)
(19,58)-(19,59)
(19,65)-(19,68)
(20,21)-(20,59)
(20,21)-(20,38)
(20,22)-(20,26)
(20,27)-(20,37)
(20,28)-(20,30)
(20,32)-(20,33)
(20,35)-(20,36)
(20,42)-(20,59)
(20,43)-(20,47)
(20,48)-(20,58)
(20,49)-(20,51)
(20,53)-(20,54)
(20,56)-(20,57)
(22,6)-(24,25)
(22,9)-(22,44)
(22,9)-(22,25)
(22,10)-(22,14)
(22,15)-(22,24)
(22,16)-(22,17)
(22,19)-(22,20)
(22,22)-(22,23)
(22,28)-(22,44)
(22,29)-(22,33)
(22,34)-(22,43)
(22,35)-(22,36)
(22,38)-(22,39)
(22,41)-(22,42)
(23,11)-(23,25)
(23,11)-(23,15)
(23,16)-(23,25)
(23,17)-(23,18)
(23,20)-(23,21)
(23,23)-(23,24)
(24,11)-(24,25)
(24,11)-(24,15)
(24,16)-(24,25)
(24,17)-(24,18)
(24,20)-(24,21)
(24,23)-(24,24)
*)
