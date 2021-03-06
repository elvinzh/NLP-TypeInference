
let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t -> (((mulByDigit i (List.rev t)) * 10) h) * i;;


(* fix

let rec mulByDigit i l =
  match List.rev l with
  | [] -> []
  | h::t ->
      (mulByDigit i (List.rev (List.map (fun x  -> x * 10) t))) @ [h * i];;

*)

(* changed spans
(5,12)-(5,50)
(5,12)-(5,54)
(5,13)-(5,47)
(5,15)-(5,25)
(5,38)-(5,39)
(5,44)-(5,46)
(5,48)-(5,49)
*)

(* type error slice
(2,3)-(5,56)
(2,19)-(5,54)
(2,21)-(5,54)
(3,2)-(5,54)
(4,10)-(4,12)
(5,12)-(5,50)
(5,12)-(5,54)
(5,13)-(5,47)
(5,14)-(5,41)
(5,15)-(5,25)
*)

(* all spans
(2,19)-(5,54)
(2,21)-(5,54)
(3,2)-(5,54)
(3,8)-(3,18)
(3,8)-(3,16)
(3,17)-(3,18)
(4,10)-(4,12)
(5,12)-(5,54)
(5,12)-(5,50)
(5,13)-(5,47)
(5,14)-(5,41)
(5,15)-(5,25)
(5,26)-(5,27)
(5,28)-(5,40)
(5,29)-(5,37)
(5,38)-(5,39)
(5,44)-(5,46)
(5,48)-(5,49)
(5,53)-(5,54)
*)
