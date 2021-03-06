
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Poly of expr* expr* expr
  | Tan of expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine a -> sin (pi *. (eval (a, x, y)))
  | Cosine a -> cos (pi *. (eval (a, x, y)))
  | Average (a,b) -> ((eval (a, x, y)) +. (eval (b, x, y))) /. 2.0
  | Times (a,b) -> (eval (a, x, y)) *. (eval (b, x, y))
  | Thresh (a,b,c,d) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (c, x, y)
      else eval (d, x, y)
  | Poly (a,b,c) ->
      (((eval (a, x, y)) *. (eval (a, x, y))) +.
         ((eval (b, x, y)) *. (eval (c, x, y))))
        / 2
  | Tan a -> (sin (pi *. (eval (a, x, y)))) /. (cos (pi *. (eval (a, x, y))));;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Poly of expr* expr* expr
  | Tan of expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine a -> sin (pi *. (eval (a, x, y)))
  | Cosine a -> cos (pi *. (eval (a, x, y)))
  | Average (a,b) -> ((eval (a, x, y)) +. (eval (b, x, y))) /. 2.0
  | Times (a,b) -> (eval (a, x, y)) *. (eval (b, x, y))
  | Thresh (a,b,c,d) ->
      if (eval (a, x, y)) < (eval (b, x, y))
      then eval (c, x, y)
      else eval (d, x, y)
  | Poly (a,b,c) ->
      (((eval (a, x, y)) *. (eval (a, x, y))) +.
         ((eval (b, x, y)) *. (eval (c, x, y))))
        /. 2.0
  | Tan a -> (sin (pi *. (eval (a, x, y)))) /. (cos (pi *. (eval (a, x, y))));;

*)

(* changed spans
(28,6)-(30,11)
(30,10)-(30,11)
*)

(* type error slice
(16,2)-(31,77)
(19,14)-(19,17)
(19,14)-(19,42)
(28,6)-(29,48)
(28,6)-(30,11)
*)

(* all spans
(13,9)-(13,26)
(13,9)-(13,12)
(13,16)-(13,26)
(13,17)-(13,21)
(13,22)-(13,25)
(15,14)-(31,77)
(16,2)-(31,77)
(16,8)-(16,9)
(17,13)-(17,14)
(18,13)-(18,14)
(19,14)-(19,42)
(19,14)-(19,17)
(19,18)-(19,42)
(19,19)-(19,21)
(19,25)-(19,41)
(19,26)-(19,30)
(19,31)-(19,40)
(19,32)-(19,33)
(19,35)-(19,36)
(19,38)-(19,39)
(20,16)-(20,44)
(20,16)-(20,19)
(20,20)-(20,44)
(20,21)-(20,23)
(20,27)-(20,43)
(20,28)-(20,32)
(20,33)-(20,42)
(20,34)-(20,35)
(20,37)-(20,38)
(20,40)-(20,41)
(21,21)-(21,66)
(21,21)-(21,59)
(21,22)-(21,38)
(21,23)-(21,27)
(21,28)-(21,37)
(21,29)-(21,30)
(21,32)-(21,33)
(21,35)-(21,36)
(21,42)-(21,58)
(21,43)-(21,47)
(21,48)-(21,57)
(21,49)-(21,50)
(21,52)-(21,53)
(21,55)-(21,56)
(21,63)-(21,66)
(22,19)-(22,55)
(22,19)-(22,35)
(22,20)-(22,24)
(22,25)-(22,34)
(22,26)-(22,27)
(22,29)-(22,30)
(22,32)-(22,33)
(22,39)-(22,55)
(22,40)-(22,44)
(22,45)-(22,54)
(22,46)-(22,47)
(22,49)-(22,50)
(22,52)-(22,53)
(24,6)-(26,25)
(24,9)-(24,44)
(24,9)-(24,25)
(24,10)-(24,14)
(24,15)-(24,24)
(24,16)-(24,17)
(24,19)-(24,20)
(24,22)-(24,23)
(24,28)-(24,44)
(24,29)-(24,33)
(24,34)-(24,43)
(24,35)-(24,36)
(24,38)-(24,39)
(24,41)-(24,42)
(25,11)-(25,25)
(25,11)-(25,15)
(25,16)-(25,25)
(25,17)-(25,18)
(25,20)-(25,21)
(25,23)-(25,24)
(26,11)-(26,25)
(26,11)-(26,15)
(26,16)-(26,25)
(26,17)-(26,18)
(26,20)-(26,21)
(26,23)-(26,24)
(28,6)-(30,11)
(28,6)-(29,48)
(28,7)-(28,45)
(28,8)-(28,24)
(28,9)-(28,13)
(28,14)-(28,23)
(28,15)-(28,16)
(28,18)-(28,19)
(28,21)-(28,22)
(28,28)-(28,44)
(28,29)-(28,33)
(28,34)-(28,43)
(28,35)-(28,36)
(28,38)-(28,39)
(28,41)-(28,42)
(29,9)-(29,47)
(29,10)-(29,26)
(29,11)-(29,15)
(29,16)-(29,25)
(29,17)-(29,18)
(29,20)-(29,21)
(29,23)-(29,24)
(29,30)-(29,46)
(29,31)-(29,35)
(29,36)-(29,45)
(29,37)-(29,38)
(29,40)-(29,41)
(29,43)-(29,44)
(30,10)-(30,11)
(31,13)-(31,77)
(31,13)-(31,43)
(31,14)-(31,17)
(31,18)-(31,42)
(31,19)-(31,21)
(31,25)-(31,41)
(31,26)-(31,30)
(31,31)-(31,40)
(31,32)-(31,33)
(31,35)-(31,36)
(31,38)-(31,39)
(31,47)-(31,77)
(31,48)-(31,51)
(31,52)-(31,76)
(31,53)-(31,55)
(31,59)-(31,75)
(31,60)-(31,64)
(31,65)-(31,74)
(31,66)-(31,67)
(31,69)-(31,70)
(31,72)-(31,73)
*)
