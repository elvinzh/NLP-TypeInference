
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
  | Sine e0 -> sin (pi *. (eval (e0, x, y)))
  | Cosine e1 -> cos (pi *. (eval (e1, x, y)))
  | Average (e2,e3) -> ((eval e2) + (eval e3)) / 2
  | Times (e4,e5) -> (eval e4) * (eval e5)
  | Thresh (e6,e7,e8,e9) ->
      if (eval e6) < (eval e7) then eval e8 else eval e9;;


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
  | Sine e0 -> sin (pi *. (eval (e0, x, y)))
  | Cosine e1 -> cos (pi *. (eval (e1, x, y)))
  | Average (e2,e3) -> ((eval (e2, x, y)) +. (eval (e3, x, y))) /. 2.0
  | Times (e4,e5) -> (eval (e4, x, y)) *. (eval (e5, x, y))
  | Thresh (e6,e7,e8,e9) ->
      if (eval (e6, x, y)) < (eval (e7, x, y))
      then eval (e8, x, y)
      else eval (e9, x, y);;

*)

(* changed spans
(19,23)-(19,46)
(19,23)-(19,50)
(19,24)-(19,33)
(19,30)-(19,32)
(19,36)-(19,45)
(19,42)-(19,44)
(19,49)-(19,50)
(20,21)-(20,30)
(20,21)-(20,42)
(20,27)-(20,29)
(20,33)-(20,42)
(20,39)-(20,41)
(22,6)-(22,56)
(22,15)-(22,17)
(22,21)-(22,30)
(22,27)-(22,29)
(22,36)-(22,43)
(22,41)-(22,43)
(22,49)-(22,56)
(22,54)-(22,56)
*)

(* type error slice
(14,2)-(22,56)
(17,15)-(17,18)
(17,15)-(17,44)
(17,26)-(17,43)
(17,27)-(17,31)
(17,32)-(17,42)
(19,23)-(19,50)
(19,24)-(19,33)
(19,25)-(19,29)
(19,30)-(19,32)
(19,36)-(19,45)
(19,37)-(19,41)
(19,42)-(19,44)
(20,21)-(20,30)
(20,21)-(20,42)
(20,22)-(20,26)
(20,27)-(20,29)
(20,33)-(20,42)
(20,34)-(20,38)
(20,39)-(20,41)
(22,9)-(22,18)
(22,10)-(22,14)
(22,15)-(22,17)
(22,21)-(22,30)
(22,22)-(22,26)
(22,27)-(22,29)
(22,36)-(22,40)
(22,36)-(22,43)
(22,41)-(22,43)
(22,49)-(22,53)
(22,49)-(22,56)
(22,54)-(22,56)
*)

(* all spans
(11,9)-(11,26)
(11,9)-(11,12)
(11,16)-(11,26)
(11,17)-(11,21)
(11,22)-(11,25)
(13,14)-(22,56)
(14,2)-(22,56)
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
(19,23)-(19,50)
(19,23)-(19,46)
(19,24)-(19,33)
(19,25)-(19,29)
(19,30)-(19,32)
(19,36)-(19,45)
(19,37)-(19,41)
(19,42)-(19,44)
(19,49)-(19,50)
(20,21)-(20,42)
(20,21)-(20,30)
(20,22)-(20,26)
(20,27)-(20,29)
(20,33)-(20,42)
(20,34)-(20,38)
(20,39)-(20,41)
(22,6)-(22,56)
(22,9)-(22,30)
(22,9)-(22,18)
(22,10)-(22,14)
(22,15)-(22,17)
(22,21)-(22,30)
(22,22)-(22,26)
(22,27)-(22,29)
(22,36)-(22,43)
(22,36)-(22,40)
(22,41)-(22,43)
(22,49)-(22,56)
(22,49)-(22,53)
(22,54)-(22,56)
*)
