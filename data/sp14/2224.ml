
type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | CosE of expr* expr* expr;;

let rec exprToString e =
  match e with
  | VarX  -> "x"
  | VarY  -> "y"
  | Sine x -> "sin(pi*" ^ ((exprToString x) ^ ")")
  | Cosine x -> "cos(pi*" ^ ((exprToString x) ^ ")")
  | Average (x1,x2) ->
      "((" ^ ((exprToString x1) ^ ("+" ^ ((exprToString x2) ^ ")/2)")))
  | Times (x1,x2) -> (exprToString x1) ^ ("*" ^ (exprToString x2))
  | Thresh (x1,x2,x3,x4) ->
      "(" ^
        ((exprToString x1) ^
           ("<" ^
              ((exprToString x2) ^
                 ("?" ^
                    ((exprToString x3) ^ (":" ^ ((exprToString x4) ^ ")")))))))
  | CosE (x1,x2,x3) ->
      "cos(pi*" ^ (x1 ^ ("*" ^ (x2 ^ (")e^(-pi*" ^ (x3 ^ "^2)")))));;


(* fix

type expr =
  | VarX
  | VarY
  | Sine of expr
  | Cosine of expr
  | Average of expr* expr
  | Times of expr* expr
  | Thresh of expr* expr* expr* expr
  | CosE of expr* expr* expr;;

let rec exprToString e =
  match e with
  | VarX  -> "x"
  | VarY  -> "y"
  | Sine x -> "sin(pi*" ^ ((exprToString x) ^ ")")
  | Cosine x -> "cos(pi*" ^ ((exprToString x) ^ ")")
  | Average (x1,x2) ->
      "((" ^ ((exprToString x1) ^ ("+" ^ ((exprToString x2) ^ ")/2)")))
  | Times (x1,x2) -> (exprToString x1) ^ ("*" ^ (exprToString x2))
  | Thresh (x1,x2,x3,x4) ->
      "(" ^
        ((exprToString x1) ^
           ("<" ^
              ((exprToString x2) ^
                 ("?" ^
                    ((exprToString x3) ^ (":" ^ ((exprToString x4) ^ ")")))))))
  | CosE (x1,x2,x3) ->
      "cos(pi*" ^
        ((exprToString x1) ^
           ("*" ^
              ((exprToString x1) ^ (")e^(-pi*" ^ ((exprToString x1) ^ "^2)")))));;

*)

(* changed spans
(29,19)-(29,21)
(29,32)-(29,34)
(29,37)-(29,64)
(29,52)-(29,54)
(29,57)-(29,62)
*)

(* type error slice
(13,2)-(29,67)
(29,18)-(29,67)
(29,19)-(29,21)
(29,22)-(29,23)
(29,31)-(29,65)
(29,32)-(29,34)
(29,35)-(29,36)
(29,51)-(29,63)
(29,52)-(29,54)
(29,55)-(29,56)
*)

(* all spans
(12,21)-(29,67)
(13,2)-(29,67)
(13,8)-(13,9)
(14,13)-(14,16)
(15,13)-(15,16)
(16,14)-(16,50)
(16,24)-(16,25)
(16,14)-(16,23)
(16,26)-(16,50)
(16,44)-(16,45)
(16,27)-(16,43)
(16,28)-(16,40)
(16,41)-(16,42)
(16,46)-(16,49)
(17,16)-(17,52)
(17,26)-(17,27)
(17,16)-(17,25)
(17,28)-(17,52)
(17,46)-(17,47)
(17,29)-(17,45)
(17,30)-(17,42)
(17,43)-(17,44)
(17,48)-(17,51)
(19,6)-(19,71)
(19,11)-(19,12)
(19,6)-(19,10)
(19,13)-(19,71)
(19,32)-(19,33)
(19,14)-(19,31)
(19,15)-(19,27)
(19,28)-(19,30)
(19,34)-(19,70)
(19,39)-(19,40)
(19,35)-(19,38)
(19,41)-(19,69)
(19,60)-(19,61)
(19,42)-(19,59)
(19,43)-(19,55)
(19,56)-(19,58)
(19,62)-(19,68)
(20,21)-(20,66)
(20,39)-(20,40)
(20,21)-(20,38)
(20,22)-(20,34)
(20,35)-(20,37)
(20,41)-(20,66)
(20,46)-(20,47)
(20,42)-(20,45)
(20,48)-(20,65)
(20,49)-(20,61)
(20,62)-(20,64)
(22,6)-(27,79)
(22,10)-(22,11)
(22,6)-(22,9)
(23,8)-(27,79)
(23,27)-(23,28)
(23,9)-(23,26)
(23,10)-(23,22)
(23,23)-(23,25)
(24,11)-(27,78)
(24,16)-(24,17)
(24,12)-(24,15)
(25,14)-(27,77)
(25,33)-(25,34)
(25,15)-(25,32)
(25,16)-(25,28)
(25,29)-(25,31)
(26,17)-(27,76)
(26,22)-(26,23)
(26,18)-(26,21)
(27,20)-(27,75)
(27,39)-(27,40)
(27,21)-(27,38)
(27,22)-(27,34)
(27,35)-(27,37)
(27,41)-(27,74)
(27,46)-(27,47)
(27,42)-(27,45)
(27,48)-(27,73)
(27,67)-(27,68)
(27,49)-(27,66)
(27,50)-(27,62)
(27,63)-(27,65)
(27,69)-(27,72)
(29,6)-(29,67)
(29,16)-(29,17)
(29,6)-(29,15)
(29,18)-(29,67)
(29,22)-(29,23)
(29,19)-(29,21)
(29,24)-(29,66)
(29,29)-(29,30)
(29,25)-(29,28)
(29,31)-(29,65)
(29,35)-(29,36)
(29,32)-(29,34)
(29,37)-(29,64)
(29,49)-(29,50)
(29,38)-(29,48)
(29,51)-(29,63)
(29,55)-(29,56)
(29,52)-(29,54)
(29,57)-(29,62)
*)
