
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Funny of expr* expr* expr
  | Funny1 of expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine a -> sin (pi *. (eval (a, x, y)))
  | Cosine b -> cos (pi *. (eval (b, x, y)))
  | Average (c,d) -> ((eval (c, x, y)) +. (eval (d, x, y))) /. 2.0
  | Times (e,f) -> (eval (e, x, y)) *. (eval (f, x, y))
  | Thresh (g,h,i,j) ->
      if (eval (g, x, y)) < (eval (h, x, y))
      then eval (i, x, y)
      else eval (j, x, y)
  | Funny (k,l,m) ->
      ((eval (k, x, y)) +. (eval (l, x, y))) +. (eval (m, x, y))
  | Funny1 n -> (sqrt (abs_float (eval (n, x, y)))) / 1000;;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | Funny of expr* expr* expr
  | Funny1 of expr;;

let pi = 4.0 *. (atan 1.0);;

let rec eval (e,x,y) =
  match e with
  | VarX  -> x
  | VarY  -> y
  | Sine a -> sin (pi *. (eval (a, x, y)))
  | Cosine b -> cos (pi *. (eval (b, x, y)))
  | Average (c,d) -> ((eval (c, x, y)) +. (eval (d, x, y))) /. 2.0
  | Times (e,f) -> (eval (e, x, y)) *. (eval (f, x, y))
  | Thresh (g,h,i,j) ->
      if (eval (g, x, y)) < (eval (h, x, y))
      then eval (i, x, y)
      else eval (j, x, y)
  | Funny (k,l,m) ->
      ((eval (k, x, y)) +. (eval (l, x, y))) +. (eval (m, x, y))
  | Funny1 n -> (sqrt (abs_float (eval (n, x, y)))) /. 1000.0;;

*)

(* changed spans
(29,16)-(29,58)
(29,54)-(29,58)
*)

(* type error slice
(16,2)-(29,58)
(19,14)-(19,17)
(19,14)-(19,42)
(29,16)-(29,51)
(29,16)-(29,58)
(29,17)-(29,21)
*)

(* all spans
(13,9)-(13,26)
(13,9)-(13,12)
(13,16)-(13,26)
(13,17)-(13,21)
(13,22)-(13,25)
(15,14)-(29,58)
(16,2)-(29,58)
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
(28,6)-(28,64)
(28,6)-(28,44)
(28,7)-(28,23)
(28,8)-(28,12)
(28,13)-(28,22)
(28,14)-(28,15)
(28,17)-(28,18)
(28,20)-(28,21)
(28,27)-(28,43)
(28,28)-(28,32)
(28,33)-(28,42)
(28,34)-(28,35)
(28,37)-(28,38)
(28,40)-(28,41)
(28,48)-(28,64)
(28,49)-(28,53)
(28,54)-(28,63)
(28,55)-(28,56)
(28,58)-(28,59)
(28,61)-(28,62)
(29,16)-(29,58)
(29,16)-(29,51)
(29,17)-(29,21)
(29,22)-(29,50)
(29,23)-(29,32)
(29,33)-(29,49)
(29,34)-(29,38)
(29,39)-(29,48)
(29,40)-(29,41)
(29,43)-(29,44)
(29,46)-(29,47)
(29,54)-(29,58)
*)
