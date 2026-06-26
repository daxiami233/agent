export default function CommandMenu({ matches, activeIndex, onChoose, onHover }) {
  if (matches.length === 0) return null;

  return (
    <div className="commandMenu">
      {matches.map((command, index) => (
        <button
          key={command.name}
          className={index === activeIndex ? "active" : ""}
          type="button"
          onClick={() => onChoose(command.name)}
          onMouseEnter={() => onHover(index)}
        >
          <span>{command.name}</span>
          <small>{command.description}</small>
        </button>
      ))}
    </div>
  );
}
