
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
  | Sine x -> sin (eval (pi *. x))
  | Cosine x -> cos (pi *. (eval x))
  | Average (x1,x2) -> ((eval x1) +. (eval x2)) /. 2.;;


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
  | Sine x1 -> sin (pi *. (eval (x1, x, y)))
  | Cosine x1 -> cos (pi *. (eval (x1, x, y)))
  | Average (x1,x2) -> ((eval (x1, x, y)) +. (eval (x2, x, y))) /. 2.;;

*)

(* changed spans
(14,2)-(19,53)
(17,18)-(17,34)
(17,19)-(17,23)
(17,31)-(17,32)
(18,16)-(18,36)
(18,33)-(18,34)
(19,23)-(19,53)
(19,30)-(19,32)
(19,37)-(19,46)
(19,43)-(19,45)
(19,51)-(19,53)
*)

(* type error slice
(13,3)-(19,55)
(13,14)-(19,53)
(14,2)-(19,53)
(17,18)-(17,34)
(17,19)-(17,23)
(17,24)-(17,33)
(17,31)-(17,32)
(18,27)-(18,35)
(18,28)-(18,32)
(18,33)-(18,34)
(19,24)-(19,33)
(19,25)-(19,29)
(19,30)-(19,32)
(19,37)-(19,46)
(19,38)-(19,42)
(19,43)-(19,45)
*)

(* all spans
(11,9)-(11,26)
(11,9)-(11,12)
(11,16)-(11,26)
(11,17)-(11,21)
(11,22)-(11,25)
(13,14)-(19,53)
(14,2)-(19,53)
(14,8)-(14,9)
(15,13)-(15,14)
(16,13)-(16,14)
(17,14)-(17,34)
(17,14)-(17,17)
(17,18)-(17,34)
(17,19)-(17,23)
(17,24)-(17,33)
(17,25)-(17,27)
(17,31)-(17,32)
(18,16)-(18,36)
(18,16)-(18,19)
(18,20)-(18,36)
(18,21)-(18,23)
(18,27)-(18,35)
(18,28)-(18,32)
(18,33)-(18,34)
(19,23)-(19,53)
(19,23)-(19,47)
(19,24)-(19,33)
(19,25)-(19,29)
(19,30)-(19,32)
(19,37)-(19,46)
(19,38)-(19,42)
(19,43)-(19,45)
(19,51)-(19,53)
*)
